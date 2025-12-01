import os
import multiprocessing
import torch
from typing import List, Any, Dict
from rainbowplus.llms.base import BaseLLM

def _vllm_worker_process(model_kwargs, device_id, input_q, output_q):
    try:
        # 1. Cô lập GPU
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        # 2. Import
        from vllm import LLM, SamplingParams
        
        # 3. Khởi tạo
        llm = LLM(**model_kwargs)
        print(f"✅ [Worker {os.getpid()}] vLLM initialized on GPU {device_id}")
        
        while True:
            task = input_q.get()
            if task is None: break
            
            req_id, method, args, kwargs = task
            try:
                result = None
                if method == 'generate':
                    query, params_dict = args
                    outputs = llm.generate([query], SamplingParams(**params_dict))
                    if params_dict.get("logprobs"):
                        result = outputs[0] 
                    else:
                        result = outputs[0].outputs[0].text if outputs else ""
                        
                elif method == 'batch_generate':
                    queries, params_dict = args
                    outputs = llm.generate(queries, SamplingParams(**params_dict))
                    
                    if params_dict.get("logprobs"):
                        result = outputs
                    else:
                        result = [o.outputs[0].text for o in outputs]
                
                output_q.put((req_id, result, None))
            except Exception as e:
                output_q.put((req_id, None, e))
                
    except Exception as e:
        print(f"❌ [Worker Error] Critical: {e}")

class vLLM(BaseLLM):
    def __init__(self, config_or_dict: Any):
        """
        Khởi tạo vLLM Wrapper.
        Hỗ trợ input là Dictionary HOẶC đối tượng LLMConfig.
        """
        # --- FIX: Xử lý thông minh đầu vào (Config Object vs Dict) ---
        if hasattr(config_or_dict, "model_kwargs"):
            # Nếu là object LLMConfig, trích xuất dict từ thuộc tính model_kwargs
            self.model_kwargs = config_or_dict.model_kwargs.copy()
        elif isinstance(config_or_dict, dict):
            # Nếu đã là dict thì copy luôn
            self.model_kwargs = config_or_dict.copy()
        else:
            # Trường hợp lạ, cố gắng ép kiểu hoặc để rỗng
            try:
                self.model_kwargs = dict(config_or_dict).copy()
            except:
                self.model_kwargs = {}
        # -------------------------------------------------------------

        self.device = self.model_kwargs.pop("device", None)
        
        # Xử lý config cũ
        if self.model_kwargs.get("tensor_parallel_size", 1) > 1 and self.device is None:
             pass 

        ctx = multiprocessing.get_context('spawn')
        self.input_queue = ctx.Queue()
        self.output_queue = ctx.Queue()
        
        self.process = ctx.Process(
            target=_vllm_worker_process,
            args=(self.model_kwargs, self.device, self.input_queue, self.output_queue),
            daemon=True 
        )
        self.process.start()
        self.req_counter = 0

    def _send_and_wait(self, method, args, kwargs=None):
        if not self.process.is_alive():
            raise RuntimeError(f"vLLM Worker {self.device} is dead!")
        self.input_queue.put((self.req_counter, method, args, kwargs or {}))
        req_id, result, error = self.output_queue.get()
        self.req_counter += 1
        if error: raise error
        return result

    def get_name(self): return self.model_kwargs.get("model", "isolated-vllm")
    
    def generate(self, query: str, sampling_params: dict): 
        return self._send_and_wait('generate', (query, sampling_params))
        
    def batch_generate(self, queries: List[str], sampling_params: dict): 
        return self._send_and_wait('batch_generate', (queries, sampling_params))

    def __del__(self):
        if hasattr(self, 'process') and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)