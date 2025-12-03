#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import json
import logging
import sys
import os
from pathlib import Path
from typing import TypeVar, List, Dict, Any
from datasets import load_dataset
from rainbowplus.switcher import LLMSwitcher
from rainbowplus.configs import ConfigurationLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def _convert_numpy_types(obj):
    """Helper function to convert numpy types to python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(i) for i in obj]
    else:
        return obj
    
def load_txt(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_json(
    file_path: str,
    field: str,
    num_samples: int = -1,
    shuffle: bool = False,
    seed: int = 0,
) -> List[str]:
    data = load_dataset("json", data_files=file_path, split="train")

    if shuffle:
        data = data.shuffle(seed=seed)

    # Determine number of samples
    sample_count = len(data) if num_samples == -1 else min(num_samples, len(data))

    return data[field][:sample_count]


def save_iteration_log(
    log_dir, adv_prompts, responses, scores, iters, timestamp, iteration=-1, max_iters=None
):
    if iteration == -1:
        log_path = log_dir / f"rainbowplus_log_{timestamp}.json"
    else:
        log_path = log_dir / f"rainbowplus_log_{timestamp}_epoch_{iteration+1}.json"

    with open(log_path, "w") as f:
        json.dump(
            {
                "max_iters": max_iters,
                "adv_prompts": {
                    str(key): value for key, value in adv_prompts._archive.items()
                },
                "responses": {
                    str(key): value for key, value in responses._archive.items()
                },
                "scores": {str(key): value for key, value in scores._archive.items()},
                "iters": {str(key): value for key, value in iters._archive.items()},
            },
            f,
            indent=2,
        )

    logger.info(f"Log saved to {log_path}")


def calculate_lexical_diversity_from_archives(all_prompts):
    flat_prompts = []
    for key, prompts in all_prompts._archive.items():
        flat_prompts.extend(prompts)
    total = len(flat_prompts)
    unique = len(set(flat_prompts))
    diversity_score = unique / total if total > 0 else 0
    return {
        'total_prompts': total,
        'unique_prompts': unique,
        'diversity_score': diversity_score
    }


def calculate_embedding_diversity_from_archives(all_prompts):
    flat_prompts = []
    for key, prompts in all_prompts._archive.items():
        flat_prompts.extend(prompts)
    if len(flat_prompts) < 2:
        return 0.0
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(flat_prompts, show_progress_bar=False)
    dists = cosine_distances(embeddings)
    n = len(flat_prompts)
    triu_indices = np.triu_indices(n, k=1)
    avg_dist = dists[triu_indices].mean() if len(triu_indices[0]) > 0 else 0.0
    return float(avg_dist)

def calculate_lexical_diversity_from_list(prompt_list):
    """Calculate lexical diversity directly from a list of strings."""
    if not prompt_list:
        return {'total_prompts': 0, 'unique_prompts': 0, 'diversity_score': 0}
    
    total = len(prompt_list)
    unique = len(set(prompt_list))
    diversity_score = unique / total if total > 0 else 0
    return {
        'total_prompts': total,
        'unique_prompts': unique,
        'diversity_score': diversity_score
    }

def calculate_embedding_diversity_from_list(prompt_list):
    """Calculate embedding diversity directly from a list of strings."""
    if not prompt_list or len(prompt_list) < 2:
        return 0.0
        
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(prompt_list, show_progress_bar=False)
    dists = cosine_distances(embeddings)
    n = len(prompt_list)
    triu_indices = np.triu_indices(n, k=1)
    avg_dist = dists[triu_indices].mean() if len(triu_indices[0]) > 0 else 0.0
    return float(avg_dist)

def save_ga_iteration_log(
    log_dir, 
    ga_instance, 
    timestamp, 
    iteration=-1, 
    max_iters=None
):
    """
    Save the current state of the Growing Archive (Centroids and Elites).
    """
    if iteration == -1:
        log_path = log_dir / f"ga_state_global_{timestamp}.json"
    else:
        log_path = log_dir / f"ga_state_{timestamp}_epoch_{iteration+1}.json"

    # 1. Prepare Elites Data
    clean_elites = {}
    for cell_id, elite_data in ga_instance.elites.items():
        clean_elites[str(cell_id)] = _convert_numpy_types(elite_data)

    # 2. Prepare Centroids Data
    active_centroids = ga_instance.centroids[:ga_instance.n_centroids]
    clean_centroids = _convert_numpy_types(active_centroids)

    # --- SỬA LỖI Ở ĐÂY: Hỗ trợ cả n_cells (cũ) và max_cells (mới) ---
    # Lấy max_cells nếu có, nếu không thì lấy n_cells, mặc định là 0
    current_capacity = getattr(ga_instance, 'max_cells', getattr(ga_instance, 'n_cells', 0))

    # 3. Construct Log Data
    log_data = {
        "meta": {
            "max_iters": max_iters,
            "iteration": iteration,
            "max_cells": current_capacity, # Đã sửa
            "n_centroids": ga_instance.n_centroids,
        },
        "centroids": clean_centroids,
        "elites": clean_elites
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    logger.info(f"GA State saved to {log_path}")

def save_ga_comprehensive_log(
    log_dir, 
    ga_instance,
    all_prompts, 
    all_responses, 
    all_scores, 
    all_similarities,
    rejection_reasons,
    timestamp, 
    iteration=-1,
    all_prompt_ids=None,
    all_parent_ids=None,
    all_seed_ids=None,
    max_iters=None
):
    """
    Save comprehensive log including BOTH history archives AND Growing Archive state.
    """
    if iteration == -1:
        log_path = log_dir / f"comprehensive_ga_log_{timestamp}.json"
    else:
        log_path = log_dir / f"comprehensive_ga_log_{timestamp}_epoch_{iteration+1}.json"

    # 1. Extract History Archives
    history_data = {
        "max_iters": max_iters,
        "all_prompts": {str(k): v for k, v in all_prompts._archive.items()},
        "all_responses": {str(k): v for k, v in all_responses._archive.items()},
        "all_scores": {str(k): v for k, v in all_scores._archive.items()},
        "all_similarities": {str(k): v for k, v in all_similarities._archive.items()},
        "rejection_reasons": {str(k): v for k, v in rejection_reasons._archive.items()},
    }

    if all_prompt_ids is not None:
        history_data["all_prompt_ids"] = {str(k): v for k, v in all_prompt_ids._archive.items()}
    if all_parent_ids is not None:
        history_data["all_parent_ids"] = {str(k): v for k, v in all_parent_ids._archive.items()}
    if all_seed_ids is not None:
        history_data["all_seed_ids"] = {str(k): v for k, v in all_seed_ids._archive.items()}

    # 2. Extract GA State
    active_centroids = ga_instance.centroids[:ga_instance.n_centroids]
    
    ga_state = {
        "n_centroids": ga_instance.n_centroids,
        "centroids": _convert_numpy_types(active_centroids),
        "elites": {}
    }
    
    # --- SỬA LỖI LOGIC MULTI-ELITE ---
    # Collect prompts specifically from GA elites for diversity calculation
    all_ga_prompts = []
    
    for cell_id, elite_data in ga_instance.elites.items():
        ga_state["elites"][str(cell_id)] = _convert_numpy_types(elite_data)
        
        # Nếu là Multi-Elite (elite_data là list)
        if isinstance(elite_data, list):
            for item in elite_data:
                if isinstance(item, dict) and "prompt" in item:
                    all_ga_prompts.append(item["prompt"])
        # Nếu là Single-Elite (elite_data là dict - code cũ)
        elif isinstance(elite_data, dict) and "prompt" in elite_data:
            all_ga_prompts.append(elite_data["prompt"])

    # 3. Combine Data
    full_log_data = {
        **history_data, 
        "ga_state": ga_state
    }

    # Calculate diversity metrics for GA elites
    try:
        diversity = calculate_lexical_diversity_from_list(all_ga_prompts) 
        embedding_diversity = calculate_embedding_diversity_from_list(all_ga_prompts)
        
        full_log_data["ga_diversity"] = diversity
        full_log_data["ga_embedding_diversity"] = embedding_diversity
    except Exception as e:
        logger.warning(f"Skipping diversity calculation in log save: {e}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(full_log_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Comprehensive GA log saved to {log_path}")

def save_attack_memory_log(log_dir, attack_memory, timestamp):
    """Save the Attack Memory to a YAML file."""
    log_path = Path(log_dir) / f"attack_memory_{timestamp}.yml"
    try:
        raw_data = attack_memory.entries_top + attack_memory.entries_bot
        clean_data = _convert_numpy_types(raw_data)
        
        with open(log_path, "w", encoding="utf-8") as f:
            yaml.dump(clean_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
        logger.info(f"Attack Memory saved to {log_path}")
    except Exception as e:
        logger.error(f"Failed to save Attack Memory: {e}")

def format_example(entry: Dict[str, Any]) -> str:
    """Chuyển đổi entry thành chuỗi YAML."""
    details = entry.get('persona', {})
    prompt_text = entry.get('prompt', '').strip()
    
    ordered_data = {}
    if prompt_text:
        ordered_data['generated_attack_prompt'] = prompt_text
    
    for key, value in details.items():
        ordered_data[key] = value

    try:
        yaml_str = yaml.dump(
            ordered_data,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=float("inf")
        )
        return yaml_str.strip()
    except Exception as e:
        return f"Error formatting persona: {e}"
    
def initialize_language_models(config: ConfigurationLoader):
    """Initialize language models from configuration."""
    model_configs = [
        config.target_llm,
        config.mutator_llm,
    ]

    llm_switchers = {}
    seen_model_configs = set()

    for model_config in model_configs:
        config_key = tuple(sorted(model_config.model_kwargs.items()))

        if config_key not in seen_model_configs:
            try:
                llm_switcher = LLMSwitcher(model_config)
                model_name = model_config.model_kwargs.get("model", "unnamed_model")
                llm_switchers[model_name] = llm_switcher
                seen_model_configs.add(config_key)
            except ValueError as e:
                logger.error(f"Error initializing model {model_config}: {e}")

    return llm_switchers