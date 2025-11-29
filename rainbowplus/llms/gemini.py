#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
from typing import List, Dict, Any

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
        """
        # Tắt bộ lọc an toàn cho Red Teaming
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        """

    def get_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, sampling_params: Dict[str, Any] = None) -> str:
        # config.sampling_params là Dictionary, dùng .get() OK
        # Sử dụng params mặc định từ config nếu không có params mới truyền vào
        default_params = self.config.sampling_params.copy() if self.config.sampling_params else {}
        
        if sampling_params:
            default_params.update(sampling_params)
            
        generation_config = genai.GenerationConfig(
            temperature=default_params.get("temperature", 0.9),
            top_p=default_params.get("top_p", 0.9),
            max_output_tokens=default_params.get("max_tokens", 512)
        )

        try:
            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            # Kiểm tra response hợp lệ
            if response.text:
                return response.text
            return ""
        except Exception as e:
            # In lỗi để debug nhưng không làm crash luồng chạy
            print(f"Gemini API Warning: {e}")
            return ""

    def batch_generate(self, prompts: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, sampling_params))
            # Nghỉ nhẹ để tránh rate limit của Google (15 request/phút với gói free)
            time.sleep(2.0) 
        return results