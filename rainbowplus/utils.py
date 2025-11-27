
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import json
import logging
import sys
import os

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
    """
    Load text file and return non-empty lines.

    Args:
        file_path (str): Path to the text file

    Returns:
        List[str]: List of stripped, non-empty lines
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_json(
    file_path: str,
    field: str,
    num_samples: int = -1,
    shuffle: bool = False,
    seed: int = 0,
) -> List[str]:
    """
    Load JSON dataset with optional sampling and shuffling.

    Args:
        file_path (str): Path to the JSON file
        field (str): Field to extract from the dataset
        num_samples (int, optional): Number of samples to return. Defaults to -1 (all).
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        seed (int, optional): Random seed for shuffling. Defaults to 0.

    Returns:
        List[str]: Extracted and potentially sampled/shuffled data
    """
    data = load_dataset("json", data_files=file_path, split="train")

    if shuffle:
        data = data.shuffle(seed=seed)

    # Determine number of samples
    sample_count = len(data) if num_samples == -1 else min(num_samples, len(data))

    return data[field][:sample_count]


def save_iteration_log(
    log_dir, adv_prompts, responses, scores, iters, timestamp, iteration=-1, max_iters=None
):
    """
    Save log of current iteration's results.

    Args:
        log_dir: Directory for saving log files
        adv_prompts: Archive of adversarial prompts
        responses: Archive of model responses
        scores: Archive of prompt scores
        timestamp: Timestamp for log filename
        iteration: Current iteration number 
            - -1: Global log
            - otherwise: Iteration log
        max_iters: Maximum iterations configured for the experiment
    """
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
    # Compute pairwise cosine distances
    dists = cosine_distances(embeddings)
    # Only take upper triangle, excluding diagonal
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
    # Compute pairwise cosine distances
    dists = cosine_distances(embeddings)
    # Only take upper triangle, excluding diagonal
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
        # Convert numpy types to native python types
        clean_elites[str(cell_id)] = _convert_numpy_types(elite_data)

    # 2. Prepare Centroids Data
    active_centroids = ga_instance.centroids[:ga_instance.n_centroids]
    clean_centroids = _convert_numpy_types(active_centroids)

    # 3. Construct Log Data
    log_data = {
        "meta": {
            "max_iters": max_iters,
            "iteration": iteration,
            "n_cells": ga_instance.n_cells,
            "n_centroids": ga_instance.n_centroids,
            "dmin": float(ga_instance.dmin) if ga_instance.dmin != np.inf else "inf"
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
    # New: Lineage Archives
    all_prompt_ids=None,
    all_parent_ids=None,
    all_seed_ids=None,
    max_iters=None
):
    """
    Save comprehensive log including BOTH history archives AND Growing Archive state.
    Compatible with Archive object structure.
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

    # Add Lineage Data if available (Extracting from Archive objects)
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
        "dmin": float(ga_instance.dmin) if ga_instance.dmin != np.inf else "inf",
        "centroids": _convert_numpy_types(active_centroids),
        "elites": {}
    }
    
    # Collect prompts specifically from GA elites for diversity calculation
    all_ga_prompts = []
    
    for cell_id, elite_data in ga_instance.elites.items():
        ga_state["elites"][str(cell_id)] = _convert_numpy_types(elite_data)
        if "prompt" in elite_data:
            all_ga_prompts.append(elite_data["prompt"])

    # 3. Combine Data
    full_log_data = {
        **history_data, 
        "ga_state": ga_state
    }

    # Calculate diversity metrics for GA elites
    try:
        # Use list-based diversity functions
        diversity = calculate_lexical_diversity_from_list(all_ga_prompts) 
        embedding_diversity = calculate_embedding_diversity_from_list(all_ga_prompts)
        
        full_log_data["ga_diversity"] = diversity
        full_log_data["ga_embedding_diversity"] = embedding_diversity
    except Exception as e:
        logger.warning(f"Skipping diversity calculation in log save: {e}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(full_log_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Comprehensive GA log saved to {log_path}")

def save_comprehensive_log(
    log_dir, 
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
    Save comprehensive log of all generated prompts including failed ones.

    Args:
        log_dir: Directory for saving log files
        all_prompts: Archive of all generated prompts (successful and failed)
        all_responses: Archive of all model responses
        all_scores: Archive of all prompt scores
        all_similarities: Archive of similarity scores
        rejection_reasons: Archive of rejection reasons for each prompt
        timestamp: Timestamp for log filename
        iteration: Current iteration number 
            - -1: Global log
            - otherwise: Iteration log
        all_prompt_ids: Archive of prompt_ids (optional)
        all_parent_ids: Archive of parent_ids (optional)
        all_seed_ids: Archive of seed_ids (optional)
        max_iters: Maximum iterations configured for the experiment
    """
    if iteration == -1:
        log_path = log_dir / f"comprehensive_log_{timestamp}.json"
    else:
        log_path = log_dir / f"comprehensive_log_{timestamp}_epoch_{iteration+1}.json"

    # Calculate lexical diversity
    diversity = calculate_lexical_diversity_from_archives(all_prompts)
    # Calculate embedding-based diversity
    embedding_diversity = calculate_embedding_diversity_from_archives(all_prompts)

    log_data = {
        "max_iters": max_iters,
        "all_prompts": {
            str(key): value for key, value in all_prompts._archive.items()
        },
        "all_responses": {
            str(key): value for key, value in all_responses._archive.items()
        },
        "all_scores": {str(key): value for key, value in all_scores._archive.items()},
        "all_similarities": {str(key): value for key, value in all_similarities._archive.items()},
        "rejection_reasons": {str(key): value for key, value in rejection_reasons._archive.items()},
        "diversity": diversity,
        "embedding_diversity": embedding_diversity,
    }
    if all_prompt_ids is not None:
        log_data["all_prompt_ids"] = {str(key): value for key, value in all_prompt_ids._archive.items()}
    if all_parent_ids is not None:
        log_data["all_parent_ids"] = {str(key): value for key, value in all_parent_ids._archive.items()}
    if all_seed_ids is not None:
        log_data["all_seed_ids"] = {str(key): value for key, value in all_seed_ids._archive.items()}

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Comprehensive log saved to {log_path}")

def _format_example(entry: Dict[str, Any]) -> str:
    """
    Chuyển đổi entry thành chuỗi YAML.
    
    Thứ tự hiển thị:
    1. PROMPT (Đứng đầu)
    2. TITLE
    3. DETAILS (Các thông tin còn lại của Persona)
    """
    # 1. Unpack dữ liệu
    persona_tuple = entry.get('persona')
    prompt_text = entry.get('prompt', '').strip()
    

    title, details = persona_tuple
    
    # 2. Tạo Dict mới (Ordered)
    ordered_data = {}

    # --- THAY ĐỔI Ở ĐÂY: Đưa Prompt vào ĐẦU TIÊN ---
    if prompt_text:
        ordered_data['generated_attack_prompt'] = prompt_text
    
    # Tiếp theo là Title
    ordered_data['title'] = title
    
    # Cuối cùng là các chi tiết Persona
    for key, value in details.items():
        ordered_data[key] = value

    # 3. Dump sang YAML string
    try:
        yaml_str = yaml.dump(
            ordered_data,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,          # Giữ nguyên thứ tự: Prompt -> Title -> Details
            width=float("inf")
        )
        return yaml_str.strip()
    except Exception as e:
        return f"Error formatting persona: {e}"
    
def initialize_language_models(config: ConfigurationLoader):
    """
    Initialize language models from configuration.

    Args:
        config: Configuration object containing model settings

    Returns:
        Dictionary of initialized language models
    """
    # Extract model configurations
    model_configs = [
        config.target_llm,
        config.mutator_llm,
    ]

    # Create unique language model switchers
    llm_switchers = {}
    seen_model_configs = set()

    for model_config in model_configs:
        # Create a hashable representation of model kwargs
        config_key = tuple(sorted(model_config.model_kwargs.items()))

        # Only create a new LLM switcher if this configuration hasn't been seen before
        if config_key not in seen_model_configs:
            try:
                llm_switcher = LLMSwitcher(model_config)
                model_name = model_config.model_kwargs.get("model", "unnamed_model")
                llm_switchers[model_name] = llm_switcher
                seen_model_configs.add(config_key)
            except ValueError as e:
                logger.error(f"Error initializing model {model_config}: {e}")

    return llm_switchers

