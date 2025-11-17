import os
import torch  # <-- 1. Import torch
from typing import List
from vllm import LLM, SamplingParams
from rainbowplus.llms.base import BaseLLM


class vLLM(BaseLLM):
    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs.copy()
        
        # --- BẮT ĐẦU CODE SỬA ĐỔI LẦN 3 ---
        
        # 2. Lấy device_id ('0' hoặc '1')
        device_id_str = self.model_kwargs.pop("device", "0")
        device_id_int = int(device_id_str)

        # 3. Lấy device hiện tại (để khôi phục sau)
        original_device_id = torch.cuda.current_device()

        try:
            # 4. Ra lệnh cho PyTorch đổi GPU active
            #    Đây là lệnh quan trọng nhất
            torch.cuda.set_device(device_id_int)
            
            # 5. Khởi tạo LLM. Giờ nó sẽ dùng GPU active (GPU 1)
            self.llm = LLM(**self.model_kwargs)
        
        finally:
            # 6. Khôi phục lại device cũ để không ảnh hưởng
            #    đến các tiến trình khác (nếu có)
            torch.cuda.set_device(original_device_id)
        
        # --- KẾT THÚC CODE SỬA ĐỔI LẦN 3 ---

    def get_name(self):
        return self.model_kwargs["model"]

    def generate(self, query: str, sampling_params: dict):
        outputs = self.llm.generate([query], SamplingParams(**sampling_params))
        response = outputs[0].outputs[0].text
        return response

    def batch_generate(self, queries: List[str], sampling_params: dict):
        outputs = self.llm.generate(queries, SamplingParams(**sampling_params))
        return [output.outputs[0].text for output in outputs]