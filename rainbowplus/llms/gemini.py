#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import random
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- THAY ĐỔI 1: Import thư viện của Colab ---
try:
    from google.colab import userdata
except ImportError:
    pass # Xử lý trường hợp không chạy trên colab nếu cần

# Giả định module này đã có trong môi trường Colab của bạn
# Nếu chưa, bạn cần: !pip install -e . hoặc chỉnh sys.path
from rainbowplus.llms.base import BaseLLM 

class GeminiLLM(BaseLLM):
    def __init__(self, config: Any):
        # config là đối tượng LLMConfig (dataclass)
        super().__init__() 
        self.config = config
        
        # --- THAY ĐỔI 2: Lấy Secret trên Colab ---
        
        
        genai.configure(api_key=config.api_key)
        
        # config.model_kwargs là Dictionary
        self.model_name = config.model_kwargs.get("model", "gemini-1.5-flash")
        
        self.model = genai.GenerativeModel(self.model_name)
        
        # Rate limit settings based on Gemini API tiers [citation:2]
        self._setup_rate_limits()
        
        # Request tracking for rate limiting
        self.request_timestamps = []
        self.total_tokens_used = 0
        self.rate_limit_reset_time = time.time() + 60  # Start with 1 minute window

    def _setup_rate_limits(self):
        """Configure rate limits based on the specific Gemini model"""
        model_limits = {
            "gemini-2.5-flash": {"rpm": 10, "tpm": 250000, "rpd": 250},  # Free tier [citation:2]
            "gemini-2.5-pro": {"rpm": 2, "tpm": 125000, "rpd": 50},     # Free tier [citation:2]
            "gemini-1.5-flash": {"rpm": 15, "tpm": 1000000, "rpd": 200}, # Free tier equivalent
        }
        
        # Default to most restrictive limits
        self.rate_limits = model_limits.get(self.model_name, {"rpm": 10, "tpm": 100000, "rpd": 100})
        logger.info(f"Configured rate limits for {self.model_name}: {self.rate_limits}")

    def _calculate_backoff(self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate exponential backoff with jitter"""
        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
        logger.info(f"Backoff attempt {attempt}: waiting {delay:.2f} seconds")
        return delay

    def _check_rate_limits(self) -> bool:
        """Check if we're approaching rate limits and need to slow down"""
        current_time = time.time()
        
        # Clean old timestamps (older than 1 minute)
        self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 60]
        
        # Check requests per minute
        if len(self.request_timestamps) >= self.rate_limits["rpm"] * 0.8:  # 80% of limit
            time_until_next = 60 - (current_time - min(self.request_timestamps))
            if time_until_next > 0:
                logger.warning(f"Approaching RPM limit. Sleeping for {time_until_next:.2f}s")
                time.sleep(time_until_next)
                return True
        
        # Reset token count if we're in a new minute
        if current_time > self.rate_limit_reset_time:
            self.total_tokens_used = 0
            self.rate_limit_reset_time = current_time + 60
        
        return False

    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4

    def get_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, sampling_params: Dict[str, Any] = None, 
                 max_retries: int = 5, base_backoff: float = 2.0) -> str:
        """
        Enhanced generate method with comprehensive rate limit handling
        """
        last_exception = None
        
        # Pre-emptive rate limit check
        self._check_rate_limits()
        
        for attempt in range(max_retries):
            try:
                # Use default params from config if no new params provided
                default_params = self.config.sampling_params.copy() if self.config.sampling_params else {}
                
                if sampling_params:
                    default_params.update(sampling_params)
                    
                generation_config = genai.GenerationConfig(
                    temperature=default_params.get("temperature", 0.9),
                    top_p=default_params.get("top_p", 0.9),
                    max_output_tokens=default_params.get("max_tokens", 512),
                    top_k=default_params.get("top_k", None),
                )

                # Track this request
                self.request_timestamps.append(time.time())
                estimated_input_tokens = self._estimate_tokens(prompt)
                self.total_tokens_used += estimated_input_tokens
                
                # Check token limits
                if self.total_tokens_used > self.rate_limits["tpm"] * 0.8:
                    wait_time = (self.rate_limit_reset_time - time.time())
                    if wait_time > 0:
                        logger.warning(f"Approaching TPM limit. Sleeping for {wait_time:.2f}s")
                        time.sleep(wait_time)
                
                response = self.model.generate_content(
                    prompt, 
                    generation_config=generation_config,
                    safety_settings=self.safety_settings
                )
                
                # Track output tokens
                if response.text:
                    output_tokens = self._estimate_tokens(response.text)
                    self.total_tokens_used += output_tokens
                
                # Validate response
                if response.text:
                    return response.text
                else:
                    raise ValueError("Empty response from Gemini API")
                    
            except (genai.types.StopCandidateException, genai.types.BlockedPromptException) as e:
                # Content safety violations - don't retry
                logger.warning(f"Content safety violation: {e}")
                return ""
            except (genai.types.RateLimitError, genai.types.InternalServerError, 
                   genai.types.ServiceUnavailableError, Exception) as e:
                last_exception = e
                logger.warning(f"Gemini API error on attempt {attempt + 1}/{max_retries}: {e}")
                
                if attempt < max_retries - 1:
                    sleep_time = self._calculate_backoff(attempt, base_backoff)
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All {max_retries} attempts failed. Last exception: {last_exception}")
        
        # If all retries fail, return empty string instead of crashing
        return ""

    def batch_generate(self, prompts: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        """
        Enhanced batch generation with intelligent pacing
        """
        results = []
        total_prompts = len(prompts)
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{total_prompts}")
            
            # Intelligent delay between requests
            if i > 0:
                # Calculate dynamic delay based on current load
                current_time = time.time()
                recent_requests = [ts for ts in self.request_timestamps if current_time - ts < 10]  # Last 10 seconds
                
                if len(recent_requests) >= self.rate_limits["rpm"] // 6:  # If more than 1/6 of RPM in last 10s
                    delay = 60 / self.rate_limits["rpm"]  # Evenly space requests
                    logger.info(f"Throttling: waiting {delay:.2f}s between requests")
                    time.sleep(delay)
                else:
                    # Minimal delay for normal operation
                    time.sleep(0.5)
            
            result = self.generate(prompt, sampling_params)
            results.append(result)
        
        return results

    @property
    def safety_settings(self) -> Dict:
        """Safety settings for red teaming - disable content filters"""
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for monitoring"""
        current_time = time.time()
        requests_last_minute = len([ts for ts in self.request_timestamps if current_time - ts < 60])
        
        return {
            "model": self.model_name,
            "requests_last_minute": requests_last_minute,
            "max_requests_per_minute": self.rate_limits["rpm"],
            "tokens_used_this_minute": self.total_tokens_used,
            "max_tokens_per_minute": self.rate_limits["tpm"],
            "time_until_reset": max(0, self.rate_limit_reset_time - current_time),
            "rate_limit_utilization": {
                "rpm": (requests_last_minute / self.rate_limits["rpm"]) * 100,
                "tpm": (self.total_tokens_used / self.rate_limits["tpm"]) * 100
            }
        }