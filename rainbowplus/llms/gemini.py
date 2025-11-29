import time
import logging
import google.generativeai as genai
from typing import Any, Dict, List # Cần thêm List cho batch_generate

# Import Exception chuẩn của Google
from google.api_core.exceptions import ResourceExhausted, InternalServerError

# Giữ nguyên phần import BaseLLM của dự án bạn

from rainbowplus.llms.base import BaseLLM

logger = logging.getLogger(__name__)

class GeminiLLM(BaseLLM):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        model_kwargs = getattr(config, "model_kwargs", {}) or {}
        
        # 1. Cấu hình API Key & Model
        self.api_key = model_kwargs.get("api_key")
        self.model_name = model_kwargs.get("model", "gemini-1.5-flash")
        
        # 2. Cấu hình Rate Limit (RPM)
        self.rpm = model_kwargs.get("rpm", 10) 
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

    # --- SỬA LỖI: Thêm hàm get_name theo yêu cầu của BaseLLM ---
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

        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit() # 1. Chờ RPM

                response = self._model.generate_content(prompt, generation_config=gen_config)
                
                self.last_call_time = time.time() # 2. Cập nhật thời gian

                if response.text:
                    return response.text
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
                logger.error(f"Generate Error: {e}")
                break
        
        return ""

    # --- SỬA LỖI: Thêm hàm batch_generate ---
    def batch_generate(self, prompts: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        """
        Xử lý danh sách prompt tuần tự để đảm bảo an toàn Rate Limit.
        """
        results = []
        total = len(prompts)
        logger.info(f"🚀 Bắt đầu batch generate cho {total} prompts với model {self.model_name}...")
        
        for i, prompt in enumerate(prompts):
            # Gọi lại hàm generate (đã có sẵn logic chờ/ngủ bên trong)
            res = self.generate(prompt, sampling_params)
            results.append(res)
            
            # Log tiến độ mỗi 10 câu để đỡ spam log
            if (i + 1) % 10 == 0:
                logger.info(f"   ...Đã xử lý {i + 1}/{total} prompts.")
                
        return results