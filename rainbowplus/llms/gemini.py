import time
import logging
import google.generativeai as genai
from typing import Any, Dict

# --- QUAN TRỌNG: Import đúng Exception class từ thư viện gốc của Google ---
from google.api_core.exceptions import ResourceExhausted, InternalServerError

try:
    from rainbowplus.llms.base import BaseLLM
except Exception:
    class BaseLLM: pass

logger = logging.getLogger(__name__)

class GeminiLLM(BaseLLM):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        model_kwargs = getattr(config, "model_kwargs", {}) or {}
        
        # 1. Cấu hình API Key
        self.api_key = model_kwargs.get("api_key")
        self.model_name = model_kwargs.get("model", "gemini-1.5-flash")
        
        # 2. Cấu hình Rate Limit (RPM - Requests Per Minute)
        # Nguồn: Google AI Studio Free Tier cho Flash là 15 RPM. 
        # Ta đặt mặc định 10 cho an toàn.
        self.rpm = model_kwargs.get("rpm", 10) 
        self.min_interval = 60.0 / float(self.rpm) # Khoảng cách tối thiểu giữa 2 lần gọi (giây)
        self.last_call_time = 0.0 # Lưu thời điểm gọi cuối cùng

        if self.api_key:
            genai.configure(api_key=self.api_key)
        else:
            logger.error("❌ ERROR: Thiếu 'api_key' trong config!")

        self._model = None

    def _ensure_model_client(self):
        if self._model is None and self.api_key:
            self._model = genai.GenerativeModel(self.model_name)

    def _wait_for_rate_limit(self):
        """
        Cơ chế chủ động: Tự động ngủ để không vượt quá RPM cho phép.
        """
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            # logger.info(f"Đang chờ {sleep_time:.2f}s để tránh Rate Limit...")
            time.sleep(sleep_time)

    def generate(self, prompt: str, sampling_params: Dict[str, Any] = None, max_retries: int = 5) -> str:
        if not self.api_key: return ""
        self._ensure_model_client()

        # Lấy tham số sampling
        default_params = dict(getattr(self.config, "sampling_params", {}) or {})
        if sampling_params: default_params.update(sampling_params)
        
        gen_config = genai.GenerationConfig(
            temperature=default_params.get("temperature", 0.7),
            max_output_tokens=default_params.get("max_tokens", 1024)
        )

        # Cơ chế Retry thông minh (Exponential Backoff)
        for attempt in range(max_retries):
            try:
                # 1. Chủ động chờ nếu gọi quá nhanh
                self._wait_for_rate_limit()

                # 2. Gọi API
                response = self._model.generate_content(prompt, generation_config=gen_config)
                
                # Cập nhật thời gian gọi thành công
                self.last_call_time = time.time()

                if response.text:
                    return response.text
                return ""

            except ResourceExhausted as e:
                # ĐÂY LÀ LỖI CHÍNH XÁC CỦA GOOGLE KHI HẾT QUOTA (429)
                wait_time = (2 ** attempt) + 2  # Ngủ: 3s, 4s, 6s, 10s...
                logger.warning(f"⚠️ Gặp Rate Limit (ResourceExhausted). Đang chờ {wait_time}s rồi thử lại...")
                time.sleep(wait_time)
                # Cập nhật lại last_call_time để lần lặp sau tính toán đúng
                self.last_call_time = time.time() 

            except InternalServerError:
                # Lỗi Server Google (500) -> Thử lại nhanh hơn
                time.sleep(2)
            
            except Exception as e:
                logger.error(f"Lỗi không xác định: {e}")
                # Với lỗi lạ thì thường không nên retry, break luôn
                break
        
        return ""