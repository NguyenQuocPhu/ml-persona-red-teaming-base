# rainbowplus/llms/gemini.py

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
from typing import List, Dict, Any
from rainbowplus.llms.base import BaseLLM  # Giả sử bạn có BaseLLM ở đây

class GeminiLLM(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        super().__init__() 
        self.config = config
        # Lấy API Key từ config hoặc biến môi trường
        api_key = config.get("api_key") 
        if not api_key:
            raise ValueError("Gemini API Key is missing in config")
        
        genai.configure(api_key=api_key)
        
        model_name = config["model_kwargs"].get("model", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(model_name)
        
        # --- QUAN TRỌNG: Cấu hình tắt Safety Filters cho Red Teaming ---
        # Nếu không tắt, Gemini sẽ từ chối sinh prompt độc hại
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
    def get_name(self) -> str:
        """Returns the name of the model."""
        return self.model_name
    
    def generate(self, prompt: str, sampling_params: Dict[str, Any] = None) -> str:
        # Merge params mặc định và params truyền vào
        params = self.config.get("sampling_params", {}).copy()
        if sampling_params:
            params.update(sampling_params)
            
        generation_config = genai.GenerationConfig(
            temperature=params.get("temperature", 0.9),
            top_p=params.get("top_p", 0.9),
            max_output_tokens=params.get("max_tokens", 512)
        )

        try:
            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            return response.text if response.text else ""
        except Exception as e:
            # Gemini có thể block hoàn toàn trả về None, hoặc lỗi mạng
            print(f"Gemini API Error: {e}")
            return ""

    def batch_generate(self, prompts: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        # Gemini API hiện tại (bản thường) không hỗ trợ batch request native như vLLM
        # Ta phải loop qua từng prompt. Có thể dùng ThreadPoolExecutor để nhanh hơn.
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, sampling_params))
            # Sleep nhẹ để tránh hit rate limit nếu dùng tier free
            time.sleep(0.5) 
        return results