import json
import argparse
import os
import glob
import yaml
from pathlib import Path
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_comprehensive_log(log_path):
    """Load a comprehensive log file."""
    with open(log_path, 'r') as f:
        return json.load(f)

def load_attack_memory_log(log_path):
    """Load an attack memory YAML file."""
    if not log_path: return None
    with open(log_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def find_log_files(directory):
    """
    Automatically find comprehensive, regular, and attack memory log files.
    """
    directory = Path(directory)
    
    # 1. Find Comprehensive Log
    comprehensive_global = directory / "comprehensive_ga_log_global.json"
    if comprehensive_global.exists():
        comprehensive_log = str(comprehensive_global)
    else:
        # Fallback pattern
        patterns = ["comprehensive_ga_log_*.json", "comprehensive_log_*.json"]
        comprehensive_log = None
        for p in patterns:
            candidates = sorted(list(glob.glob(str(directory / p))))
            if candidates:
                comprehensive_log = candidates[-1]
                break
    
    # 2. Find Regular Log (GA State)
    regular_log = None
    patterns = ["ga_state_global.json", "ga_state_*.json", "rainbowplus_log_*.json"]
    for p in patterns:
        candidates = sorted(list(glob.glob(str(directory / p))))
        if candidates:
            regular_log = candidates[-1]
            break
            
    # 3. Find Attack Memory Log (NEW)
    attack_memory_log = None
    memory_candidates = sorted(list(glob.glob(str(directory / "attack_memory_*.yml"))))
    if memory_candidates:
        attack_memory_log = memory_candidates[-1]

    
    return comprehensive_log, regular_log, attack_memory_log

def calculate_self_bleu(log_data, n=4):
    """
    Calculate Self-BLEU for the set of prompts.
    Lower is more diverse.
    """
    ga_state = log_data.get('ga_state', {})
    elites = ga_state.get('elites', {})
    flat_prompts = []
    for key, elite in elites.items():
        prompt_text = elite.get('prompt', "")
        if prompt_text: flat_prompts.append(prompt_text)
    if len(flat_prompts) < 2:
        return 0.0

    # Tokenize prompts
    tokenized_prompts = [p.split() for p in flat_prompts]
    smoother = SmoothingFunction().method1
    scores = []
    for i, candidate in enumerate(tokenized_prompts):
        references = tokenized_prompts[:i] + tokenized_prompts[i+1:]
        if not references:
            continue
        score = sentence_bleu(references, candidate, weights=tuple([1/n]*n), smoothing_function=smoother)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0
