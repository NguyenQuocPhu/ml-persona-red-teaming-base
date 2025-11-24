#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import json
import logging
import sys
import os
from copy import deepcopy

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
    sample_count = len(data) if num_samples == -1 else min(num_samples, len(data))
    return data[field][:sample_count]

def initialize_language_models(config: ConfigurationLoader):
    model_configs = [config.target_llm, config.mutator_llm]
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

# --- DIVERSITY CALCULATION FUNCTIONS ---

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
    # Only take upper triangle, excluding diagonal
    n = len(prompt_list)
    triu_indices = np.triu_indices(n, k=1)
    avg_dist = dists[triu_indices].mean() if len(triu_indices[0]) > 0 else 0.0
    return float(avg_dist)

# --- LOGGING FUNCTIONS FOR GROWING ARCHIVE ---

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