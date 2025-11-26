import os
from typing import List
from vllm import LLM, SamplingParams
from rainbowplus.llms.base import BaseLLM

class vLLM(BaseLLM):
    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs.copy()
        
        # 1. Kiểm tra xem user có muốn chạy song song nhiều GPU không
        tp_size = self.model_kwargs.get("tensor_parallel_size", 1)
        
        # Lấy device từ config (nếu có), nếu không có thì mặc định xử lý sau
        device_id = self.model_kwargs.pop("device", None)

        original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # --- LOGIC SỬA ĐỔI THÔNG MINH ---
        if tp_size > 1:
            # TRƯỜNG HỢP 1: Chạy đa GPU (Fitness Model Llama-Guard-3-8B)
            # Không được set CUDA_VISIBLE_DEVICES thành 1 số đơn lẻ.
            # Ta để nguyên cho vLLM tự nhìn thấy tất cả GPU, hoặc set là "0,1"
            print(f"Detected Tensor Parallelism = {tp_size}. Exposing ALL GPUs.")
            # Nếu muốn chắc chắn, có thể ép nhìn thấy cả 2:
            # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
            pass 
        else:
            # TRƯỜNG HỢP 2: Chạy đơn GPU (Target/Mutator cũ)
            # Vẫn giữ logic cũ để cô lập GPU nếu cần
            if device_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        try:
            # Khởi tạo vLLM
            self.llm = LLM(**self.model_kwargs)
        finally:
            # Khôi phục trạng thái môi trường cũ
            if original_visible_devices is None:
                if "CUDA_VISIBLE_DEVICES" in os.environ and tp_size == 1:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices

    def get_name(self):
        return self.model_kwargs["model"]

    def generate(self, query: str, sampling_params: dict):
        outputs = self.llm.generate([query], SamplingParams(**sampling_params))
        response = outputs[0].outputs[0].text
        return response

    def batch_generate(self, queries: List[str], sampling_params: dict):
        outputs = self.llm.generate(queries, SamplingParams(**sampling_params))
        return [output.outputs[0].text for output in outputs]