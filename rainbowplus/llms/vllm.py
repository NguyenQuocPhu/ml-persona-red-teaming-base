import os
from typing import List
from vllm import LLM, SamplingParams
from rainbowplus.llms.base import BaseLLM


class vLLM(BaseLLM):
    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs.copy()
        
        # --- BẮT ĐẦU PHẦN CODE SỬA ĐỔI ---
        # 1. Lấy device từ config, mặc định là '0' (GPU đầu tiên)
        device_id = self.model_kwargs.pop("device", "0")

        # 2. Lấy biến môi trường CUDA_VISIBLE_DEVICES hiện tại (nếu có)
        original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # 3. TẠM THỜI "bịt mắt" script, chỉ cho nó thấy GPU chúng ta muốn
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        try:
            # 4. Khởi tạo vLLM. Bây giờ nó sẽ tự động chọn GPU duy nhất nó thấy
            self.llm = LLM(**self.model_kwargs)
        finally:
            # 5. Khôi phục biến môi trường về trạng thái ban đầu
            #    để không ảnh hưởng đến các model khác.
            if original_visible_devices is None:
                if "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices
        # --- KẾT THÚC PHẦN CODE SỬA ĐỔI ---

    def get_name(self):
        return self.model_kwargs["model"]

    def generate(self, query: str, sampling_params: dict):
        outputs = self.llm.generate([query], SamplingParams(**sampling_params))
        response = outputs[0].outputs[0].text
        return response

    def batch_generate(self, queries: List[str], sampling_params: dict):
        outputs = self.llm.generate(queries, SamplingParams(**sampling_params))
        return [output.outputs[0].text for output in outputs]