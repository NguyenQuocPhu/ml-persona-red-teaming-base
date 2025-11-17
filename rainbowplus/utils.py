def initialize_language_models(config: ConfigurationLoader):
    """
    Initialize language models from configuration.

    Args:
        config: Configuration object containing model settings

    Returns:
        Dictionary of initialized language models
    """

    # --- BẮT ĐẦU SỬA ĐỔI ---
    # 1. Lưu lại biến môi trường gốc
    original_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    logger.info(f"Original CUDA_VISIBLE_DEVICES: {original_visible_devices}")
    # --- KẾT THÚC SỬA ĐỔI ---

    # Extract model configurations
    model_configs = [
        config.target_llm,
        config.mutator_llm,
        config.fitness_llm  # <-- RẤT QUAN TRỌNG: Thêm model thứ 3 còn thiếu
    ]

    # Create unique language model switchers
    llm_switchers = {}
    seen_model_configs = set()

    # --- BẮT ĐẦU SỬA ĐỔI ---
    try:  # Dùng try/finally để đảm bảo biến môi trường được khôi phục
        for model_config in model_configs:
            
            # 2. Lấy device_id từ config của model này
            device_id = model_config.model_kwargs.get("device", "0")
            model_name_for_log = model_config.model_kwargs.get("model", "unnamed")

            # 3. "Bịt mắt" tiến trình, chỉ cho nó thấy GPU này
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            logger.info(f"Temporarily setting CUDA_VISIBLE_DEVICES={device_id} for loading model: {model_name_for_log}")

            # Create a hashable representation of model kwargs
            config_key = tuple(sorted(model_config.model_kwargs.items()))

            # Only create a new LLM switcher if this configuration hasn't been seen before
            if config_key not in seen_model_configs:
                try:
                    # 4. Bây giờ, LLMSwitcher và vLLM sẽ chỉ thấy GPU {device_id}
                    llm_switcher = LLMSwitcher(model_config)
                    model_name = model_config.model_kwargs.get("model", "unnamed_model")
                    
                    if model_name in llm_switchers:
                        # Xử lý trường hợp tên model bị trùng (ví dụ: nếu bạn dùng cùng 1 model cho mutator và fitness)
                        model_name = f"{model_name}_device_{device_id}_{len(llm_switchers)}"

                    llm_switchers[model_name] = llm_switcher
                    seen_model_configs.add(config_key)
                
                except Exception as e:
                    # Bắt lỗi chung, như OOM
                    logger.error(f"FATAL Error initializing model {model_name_for_log} on (masked) device {device_id}: {e}")
                    # Ném lại lỗi để script dừng
                    raise e
            
            logger.info(f"Successfully loaded model: {model_name_for_log}")

    finally:
        # 5. Khôi phục biến môi trường về trạng thái ban đầu
        logger.info(f"Restoring CUDA_VISIBLE_DEVICES to original value: {original_visible_devices}")
        if original_visible_devices is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_visible_devices
    # --- KẾT THÚC SỬA ĐỔI ---

    return llm_switchers