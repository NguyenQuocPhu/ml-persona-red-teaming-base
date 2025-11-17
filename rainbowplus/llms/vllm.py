import os
from typing import List
from vllm import LLM, SamplingParams
from vllm.config import DeviceConfig  # <-- 1. Import class cấu hình
from rainbowplus.llms.base import BaseLLM


class vLLM(BaseLLM):
    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs.copy()
        
        # --- BẮT ĐẦU CODE SỬA ĐỔI MỚI ---
        
        # 2. Lấy device_id ('0' hoặc '1') từ config của bạn
        device_id_str = self.model_kwargs.pop("device", "0")

        # 3. vLLM yêu cầu một list các ID, nên chúng ta chuyển '1' -> [1]
        try:
            device_ids_list = [int(device_id_str)]
        except ValueError:
            # Xử lý dự phòng nếu config là "0,1" (dù bạn không dùng)
            device_ids_list = [int(d.strip()) for d in device_id_str.split(',')]

        # 4. Tạo đối tượng DeviceConfig mà vLLM hiểu
        device_config = DeviceConfig(device_type="cuda",
                                     device_ids=device_ids_list)

        # 5. Thêm cấu hình này vào kwargs để vLLM sử dụng
        self.model_kwargs['device_config'] = device_config
        
        # 6. Khởi tạo LLM. Giờ nó sẽ dùng đúng GPU được chỉ định.
        self.llm = LLM(**self.model_kwargs)
        
        # --- KẾT THÚC CODE SỬA ĐỔI MỚI ---

    def get_name(self):
        return self.model_kwargs["model"]

    def generate(self, query: str, sampling_params: dict):
        outputs = self.llm.generate([query], SamplingParams(**sampling_params))
        response = outputs[0].outputs[0].text
        return response

    def batch_generate(self, queries: List[str], sampling_params: dict):
        outputs = self.llm.generate(queries, SamplingParams(**sampling_params))
        return [output.outputs[0].text for output in outputs]