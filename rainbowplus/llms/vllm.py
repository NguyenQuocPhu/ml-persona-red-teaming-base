import os
import multiprocessing
from typing import List, Any, Dict
# Lưu ý: KHÔNG import vllm ở đây để tránh khởi tạo CUDA sai ở main process
from rainbowplus.llms.base import BaseLLM

# --- HÀM WORKER CHẠY TRONG SANDBOX RIÊNG BIỆT ---
def _vllm_worker_process(model_kwargs, device_id, input_q, output_q):
    """
    Hàm này chạy trong một Process hoàn toàn mới.
    Tại đây ta có thể set biến môi trường an toàn tuyệt đối.
    """
    try:
        # 1. THIẾT LẬP MÔI TRƯỜNG CÔ LẬP GPU
        # Bước này phải làm TRƯỚC KHI import vllm/torch
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            print(f"🔒 [Child Process] Masked GPU: Visible Devices = {device_id}")
        
        # 2. Bây giờ mới import vLLM (để nó nhận diện môi trường mới)
        from vllm import LLM, SamplingParams
        
        # 3. Khởi tạo Engine
        llm = LLM(**model_kwargs)
        print(f"✅ [Child Process] vLLM initialized successfully on {device_id}")
        
        # 4. Vòng lặp lắng nghe và xử lý yêu cầu
        while True:
            task = input_q.get()
            if task is None: # Tín hiệu dừng
                break
            
            req_id, method, args = task
            
            try:
                result = None
                if method == 'generate':
                    # args: (query, sampling_params)
                    query, params_dict = args
                    outputs = llm.generate([query], SamplingParams(**params_dict))
                    if outputs:
                        result = outputs[0].outputs[0].text
                    else:
                        result = ""
                        
                elif method == 'batch_generate':
                    # args: (queries, sampling_params)
                    queries, params_dict = args
                    outputs = llm.generate(queries, SamplingParams(**params_dict))
                    result = [o.outputs[0].text for o in outputs]
                
                output_q.put((req_id, result, None)) # (ID, Data, Error)
                
            except Exception as inner_e:
                output_q.put((req_id, None, inner_e))
                
    except Exception as e:
        # Nếu khởi tạo thất bại, gửi lỗi về main process
        # Dùng vòng lặp để xả queue nếu cần, nhưng ở đây ta gửi lỗi fatal
        # Lưu ý: Main process cần cơ chế timeout để không bị treo nếu worker chết sớm
        print(f"❌ [Child Process] Critical Error: {e}")
        pass

# --- CLASS WRAPPER CHÍNH ---
class vLLM(BaseLLM):
    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs.copy()
        
        # Lấy device từ config (VD: "0" hoặc "1")
        self.device = self.model_kwargs.pop("device", None)
        
        # Kiểm tra Tensor Parallel (để xử lý Fitness Model chạy nhiều GPU)
        tp_size = self.model_kwargs.get("tensor_parallel_size", 1)
        if tp_size > 1 and self.device is None:
            # Nếu chạy TP mà không chỉ định device, ta giả định dùng tất cả
            # Hoặc bạn có thể set self.device = "0,1" trong config
            pass

        # Sử dụng 'spawn' context để đảm bảo process mới sạch sẽ (quan trọng cho CUDA)
        ctx = multiprocessing.get_context('spawn')
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        
        print(f"🚀 [Main Process] Spawning isolated vLLM worker for device: {self.device}")
        
        self.process = ctx.Process(
            target=_vllm_worker_process,
            args=(self.model_kwargs, self.device, self.input_queue, self.output_queue)
        )
        self.process.start()
        
        # Biến đếm request để map kết quả trả về
        self.req_counter = 0

    def _send_and_wait(self, method, args):
        """Gửi yêu cầu sang process con và đợi kết quả"""
        if not self.process.is_alive():
            raise RuntimeError("vLLM Worker Process is dead!")
            
        req_id = self.req_counter
        self.req_counter += 1
        
        self.input_queue.put((req_id, method, args))
        
        # Đợi kết quả (Blocking)
        r_id, result, error = self.output_queue.get()
        
        if error:
            raise error
        return result

    def get_name(self):
        return self.model_kwargs.get("model", "isolated-vllm")

    def generate(self, query: str, sampling_params: dict):
        return self._send_and_wait('generate', (query, sampling_params))

    def batch_generate(self, queries: List[str], sampling_params: dict):
        return self._send_and_wait('batch_generate', (queries, sampling_params))

    def __del__(self):
        # Dọn dẹp process khi object bị hủy
        if hasattr(self, 'process') and self.process.is_alive():
            self.input_queue.put(None) # Gửi tín hiệu dừng
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()