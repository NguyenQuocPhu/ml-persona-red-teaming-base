import time
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import ResourceExhausted, InternalServerError
from typing import Any, Dict, List

# Giữ nguyên import BaseLLM
try:
    from rainbowplus.llms.base import BaseLLM
except ImportError:
    class BaseLLM: 
        def get_name(self): pass
        def batch_generate(self): pass

logger = logging.getLogger(__name__)

class GeminiLLM(BaseLLM):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        model_kwargs = getattr(config, "model_kwargs", {}) or {}
        
        # 1. Cấu hình API Key & Model
        self.api_key = model_kwargs.get("api_key")
        # Tự động xóa prefix 'models/' nếu có
        raw_model_name = model_kwargs.get("model", "gemini-1.5-flash")
        self.model_name = raw_model_name.replace("models/", "")
        
        # 2. Cấu hình Rate Limit (RPM)
        self.rpm = model_kwargs.get("rpm", 5) 
        self.min_interval = 60.0 / float(self.rpm)
        self.last_call_time = 0.0

        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            logger.error("❌ ERROR: Thiếu 'api_key' trong config!")

        self._model = None

    def _ensure_model_client(self):
        if self._model is None and self.api_key:
            self._model = genai.GenerativeModel(self.model_name)

    def get_name(self) -> str:
        return self.model_name

    def _wait_for_rate_limit(self):
        """Chủ động ngủ để đảm bảo không vượt quá RPM"""
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)

    def generate(self, prompt: str, sampling_params: Dict[str, Any] = None, max_retries: int = 5) -> str:
        if not self.api_key: return ""
        self._ensure_model_client()

        default_params = dict(getattr(self.config, "sampling_params", {}) or {})
        if sampling_params: default_params.update(sampling_params)
        
        gen_config = genai.GenerationConfig(
            temperature=default_params.get("temperature", 0.7),
            max_output_tokens=default_params.get("max_tokens", 1024),
            top_p=default_params.get("top_p", 0.9)
        )

        # --- QUAN TRỌNG: Tắt bộ lọc an toàn (BLOCK_NONE) ---
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        # ---------------------------------------------------

        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit() # 1. Chờ RPM

                # Gọi API với safety_settings
                response = self._model.generate_content(
                    prompt, 
                    generation_config=gen_config,
                    safety_settings=safety_settings
                )
                
                self.last_call_time = time.time() # 2. Cập nhật thời gian

                # Kiểm tra phản hồi an toàn
                if response.candidates and response.candidates[0].content.parts:
                    return response.text
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.warning(f"⚠️ Prompt blocked by Google: {response.prompt_feedback.block_reason}")
                    return "I cannot answer." # Trả về text giả để code không crash
                else:
                    return ""

            except ResourceExhausted:
                # Lỗi 429: Quá tải -> Ngủ lâu hơn
                wait_time = (2 ** attempt) + 2
                logger.warning(f"⚠️ Rate Limit (429). Waiting {wait_time}s...")
                time.sleep(wait_time)
                self.last_call_time = time.time()

            except InternalServerError:
                time.sleep(2)
            
            except Exception as e:
                # Bắt lỗi Invalid operation (nếu safety filter vẫn lọt lưới)
                if "Invalid operation" in str(e) or "finish_reason" in str(e):
                    logger.warning("⚠️ Safety Filter Blocked (Finish Reason 2). Returning empty.")
                    return "I cannot answer."
                
                logger.error(f"Generate Error: {e}")
                break
        
        return ""

    def batch_generate(self, prompts: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        """
        Xử lý danh sách prompt tuần tự để đảm bảo an toàn Rate Limit.
        """
        results = []
        total = len(prompts)
        logger.info(f"🚀 Bắt đầu batch generate cho {total} prompts với model {self.model_name}...")
        
        for i, prompt in enumerate(prompts):
            res = self.generate(prompt, sampling_params)
            results.append(res)
            
            if (i + 1) % 10 == 0:
                logger.info(f"   ...Đã xử lý {i + 1}/{total} prompts.")
                
        return results