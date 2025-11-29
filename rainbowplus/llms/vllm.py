import os
import multiprocessing
import time
import signal
from typing import List
from rainbowplus.llms.base import BaseLLM

def _vllm_worker_process(model_kwargs, device_id, input_q, output_q):
    try:
        # 1. Cô lập GPU
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        # 2. Import vLLM sau khi set biến môi trường
        from vllm import LLM, SamplingParams
        import torch
        
        # 3. Dọn dẹp rác CUDA trước khi load (nếu có)
        torch.cuda.empty_cache()

        # 4. Khởi tạo
        llm = LLM(**model_kwargs)
        print(f"✅ [Worker {os.getpid()}] vLLM initialized on GPU {device_id}")
        
        while True:
            task = input_q.get()
            if task is None: break # Tín hiệu dừng
            
            req_id, method, args = task
            try:
                result = None
                if method == 'generate':
                    query, params_dict = args
                    outputs = llm.generate([query], SamplingParams(**params_dict))
                    result = outputs[0].outputs[0].text if outputs else ""
                elif method == 'batch_generate':
                    queries, params_dict = args
                    outputs = llm.generate(queries, SamplingParams(**params_dict))
                    result = [o.outputs[0].text for o in outputs]
                
                output_q.put((req_id, result, None))
            except Exception as e:
                output_q.put((req_id, None, e))
                
    except Exception as e:
        print(f"❌ [Worker Error] Critical: {e}")
        # Gửi lỗi về main process nếu khởi tạo thất bại
        # (Cần cơ chế handshake tốt hơn trong thực tế, nhưng tạm thời print ra log)

class vLLM(BaseLLM):
    def __init__(self, model_kwargs: dict):
        self.model_kwargs = model_kwargs.copy()
        self.device = self.model_kwargs.pop("device", None)
        
        # Xử lý config cũ
        if self.model_kwargs.get("tensor_parallel_size", 1) > 1 and self.device is None:
             pass 

        ctx = multiprocessing.get_context('spawn')
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        
        # --- DAEMON = TRUE: Process con sẽ chết ngay khi Main chết ---
        self.process = ctx.Process(
            target=_vllm_worker_process,
            args=(self.model_kwargs, self.device, self.input_queue, self.output_queue),
            daemon=True 
        )
        self.process.start()
        self.req_counter = 0

    def _send_and_wait(self, method, args):
        if not self.process.is_alive():
            raise RuntimeError(f"vLLM Worker {self.device} is dead! Check logs for OOM.")
        self.input_queue.put((self.req_counter, method, args))
        req_id, result, error = self.output_queue.get()
        self.req_counter += 1
        if error: raise error
        return result

    def get_name(self): return self.model_kwargs.get("model", "isolated-vllm")
    def generate(self, q, p): return self._send_and_wait('generate', (q, p))
    def batch_generate(self, q, p): return self._send_and_wait('batch_generate', (q, p))

    def __del__(self):
        if hasattr(self, 'process') and self.process.is_alive():
            self.process.terminate() # Giết ngay lập tức
            self.process.join(timeout=1)