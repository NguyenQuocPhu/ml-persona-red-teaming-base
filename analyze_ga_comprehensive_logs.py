#!/usr/bin/env python3

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Analysis script for comprehensive GA logs from RainbowPlus.
Full version with all helper functions included.
"""

import json
import argparse
import os
import glob
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


def find_log_files(directory):
    """
    Automatically find comprehensive GA logs and regular logs.
    """
    directory = Path(directory)
    
    # 1. Look for GA specific logs first (New Format)
    comprehensive_global = directory / "comprehensive_ga_log_global.json"
    
    if comprehensive_global.exists():
        comprehensive_log = str(comprehensive_global)
    else:
        # Find timestamped comprehensive logs with GA prefix
        comprehensive_pattern = directory / "comprehensive_ga_log_*.json"
        comprehensive_candidates = list(glob.glob(str(comprehensive_pattern)))
        
        # Fallback to old naming if GA logs not found
        if not comprehensive_candidates:
            comprehensive_pattern = directory / "comprehensive_log_*.json"
            comprehensive_candidates = list(glob.glob(str(comprehensive_pattern)))

        if not comprehensive_candidates:
            raise FileNotFoundError(f"No comprehensive log files found in {directory}")
        
        # Sort by timestamp and take the latest
        comprehensive_log = sorted(comprehensive_candidates)[-1]
    
    # 2. Find regular log file (for legacy compatibility)
    regular_log = None
    regular_patterns = [
        "ga_state_global_*.json", # Prefer GA state files
        "rainbowplus_log_*.json"
    ]
    
    for pattern in regular_patterns:
        candidates = list(glob.glob(str(directory / pattern)))
        if candidates:
            regular_log = sorted(candidates)[-1]
            break
            
    # 3. Extract max_iters
    max_iters_from_logs = None
    try:
        with open(comprehensive_log, 'r') as f:
            comp_data = json.load(f)
            # Check meta first (new structure), then root (old structure)
            if 'meta' in comp_data:
                max_iters_from_logs = comp_data['meta'].get('max_iters')
            else:
                max_iters_from_logs = comp_data.get('max_iters')
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    
    return comprehensive_log, regular_log, max_iters_from_logs


def analyze_rejection_patterns(log_data):
    """Analyze rejection patterns including GA-specific statuses."""
    rejection_reasons = log_data.get('rejection_reasons', {})
    all_scores = log_data.get('all_scores', {})
    all_similarities = log_data.get('all_similarities', {})
    
    all_reasons = []
    for key, reasons in rejection_reasons.items():
        all_reasons.extend(reasons)
    
    reason_counts = Counter(all_reasons)
    total_prompts = len(all_reasons)
    
    # GA Successful statuses
    ga_success_statuses = {
        'added_new_niche', 
        'replaced_novelty', 
        'replaced_fitness', 
        'replaced_fitness_improvement',
        'accepted'
    }
    accepted_count = sum(reason_counts[s] for s in ga_success_statuses if s in reason_counts)
    
    # Filter rejections
    similarity_rejected = reason_counts.get('similarity_too_high', 0)
    fitness_rejected = reason_counts.get('fitness_too_low', 0) + reason_counts.get('rejected_fitness_low', 0)
    ga_rejected = reason_counts.get('rejected_not_elite', 0) 
    
    success_rate = accepted_count / total_prompts if total_prompts > 0 else 0
    
    # Score stats helper
    def get_flat_values(archive_dict):
        values = []
        for key in archive_dict:
            values.extend(archive_dict[key])
        return values

    all_score_values = get_flat_values(all_scores)
    all_similarity_values = get_flat_values(all_similarities)
    
    def get_stats(values):
        if not values: return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }

    return {
        'total_prompts': total_prompts,
        'accepted_count': accepted_count,
        'similarity_rejected': similarity_rejected,
        'fitness_rejected': fitness_rejected,
        'ga_rejected': ga_rejected,
        'success_rate': success_rate,
        'reason_counts': dict(reason_counts),
        'score_stats': get_stats(all_score_values),
        'similarity_stats': get_stats(all_similarity_values),
    }


def analyze_ga_state(log_data):
    """Analyze the internal state of the Growing Archive if present."""
    if 'ga_state' not in log_data:
        return None
        
    ga_state = log_data['ga_state']
    n_centroids = ga_state.get('n_centroids', 0)
    dmin = ga_state.get('dmin', 0)
    elites = ga_state.get('elites', {})
    
    # Extract prompts from elites to calculate diversity of the survivors
    elite_prompts = []
    for elite_data in elites.values():
        if isinstance(elite_data, dict) and 'prompt' in elite_data:
            elite_prompts.append(elite_data['prompt'])
            
    # Reuse calculation function
    lexical = calculate_lexical_diversity({'temp': elite_prompts})
    
    return {
        'n_centroids': n_centroids,
        'dmin': dmin,
        'n_elites': len(elites),
        'elite_diversity': lexical['diversity_score']
    }


def analyze_by_category(log_data):
    """Analyze rejection patterns by category."""
    rejection_reasons = log_data.get('rejection_reasons', {})
    all_scores = log_data.get('all_scores', {})
    
    category_analysis = {}
    
    success_statuses = {'accepted', 'added_new_niche', 'replaced_novelty', 'replaced_fitness', 'replaced_fitness_improvement'}
    
    for key, reasons in rejection_reasons.items():
        try:
            # Handle tuple string representation
            category = eval(key)[0] if isinstance(key, str) else key[0]
        except:
            category = "unknown"
        
        if category not in category_analysis:
            category_analysis[category] = {'total': 0, 'accepted': 0, 'scores': []}
        
        for i, reason in enumerate(reasons):
            category_analysis[category]['total'] += 1
            if reason in success_statuses:
                category_analysis[category]['accepted'] += 1
                
        if key in all_scores:
            category_analysis[category]['scores'].extend(all_scores[key])
    
    for cat, data in category_analysis.items():
        data['success_rate'] = data['accepted'] / data['total'] if data['total'] > 0 else 0
        data['avg_score'] = np.mean(data['scores']) if data['scores'] else 0
        
    return category_analysis


def calculate_lexical_diversity(log_data):
    # Handle both archive structure and list input (via temp helper)
    if 'all_prompts' in log_data:
        all_prompts = log_data['all_prompts']
        flat_prompts = []
        for key, prompts in all_prompts.items():
            flat_prompts.extend(prompts)
    elif 'temp' in log_data: 
        flat_prompts = log_data['temp']
    else:
        return {'diversity_score': 0, 'unique_prompts': 0, 'total_prompts': 0}

    total = len(flat_prompts)
    unique = len(set(flat_prompts))
    diversity_score = unique / total if total > 0 else 0
    return {
        'total_prompts': total,
        'unique_prompts': unique,
        'diversity_score': diversity_score
    }


def calculate_embedding_diversity(log_data):
    # Extract prompts
    if 'all_prompts' in log_data:
        all_prompts = log_data['all_prompts']
        flat_prompts = []
        for key, prompts in all_prompts.items():
            flat_prompts.extend(prompts)
    else:
        return 0.0
    
    if len(flat_prompts) < 2:
        return 0.0
    
    # Optimization: Limit samples for speed
    if len(flat_prompts) > 1000:
        flat_prompts = np.random.choice(flat_prompts, 1000, replace=False)

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(flat_prompts, show_progress_bar=False)
        dists = cosine_distances(embeddings)
        n = len(flat_prompts)
        triu_indices = np.triu_indices(n, k=1)
        avg_dist = dists[triu_indices].mean() if len(triu_indices[0]) > 0 else 0.0
        return float(avg_dist)
    except Exception as e:
        print(f"Warning: Embedding diversity calc failed: {e}")
        return 0.0


def calculate_self_bleu(log_data, n=4):
    if 'all_prompts' in log_data:
        all_prompts = log_data['all_prompts']
        flat_prompts = []
        for key, prompts in all_prompts.items():
            flat_prompts.extend(prompts)
    else:
        return 0.0
    
    if len(flat_prompts) < 2: return 0.0
    
    # Optimization: Limit samples
    if len(flat_prompts) > 500:
        flat_prompts = np.random.choice(flat_prompts, 500, replace=False)

    tokenized = [p.split() for p in flat_prompts]
    smoother = SmoothingFunction().method1
    scores = []
    for i, candidate in enumerate(tokenized):
        references = tokenized[:i] + tokenized[i+1:]
        if not references: continue
        score = sentence_bleu(references, candidate, weights=tuple([1/n]*n), smoothing_function=smoother)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def print_analysis(analysis_results, category_analysis, ga_stats=None):
    print("=" * 60)
    print("COMPREHENSIVE GA LOG ANALYSIS")
    print("=" * 60)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total prompts: {analysis_results['total_prompts']}")
    print(f"GA Accepted (Success): {analysis_results['accepted_count']}")
    print(f"Rejected (Similarity): {analysis_results['similarity_rejected']}")
    print(f"Rejected (Fitness): {analysis_results['fitness_rejected']}")
    print(f"Rejected (GA Density): {analysis_results['ga_rejected']}")
    print(f"Success rate: {analysis_results['success_rate']:.2%}")
    
    if ga_stats:
        print(f"\nGROWING ARCHIVE STATE:")
        print(f"Active Centroids: {ga_stats['n_centroids']}")
        print(f"Threshold (dmin): {ga_stats['dmin']:.4f}")
        print(f"Elite Population: {ga_stats['n_elites']}")
        print(f"Survivor Diversity: {ga_stats['elite_diversity']:.3f}")

    print(f"\nCATEGORY ANALYSIS:")
    for category, data in category_analysis.items():
        print(f"  {category}: Total {data['total']}, Accepted {data['accepted']} ({data['success_rate']:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Analyze comprehensive GA logs")
    parser.add_argument("directory", help="Directory containing log files")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--max-iterations", type=int)
    
    args = parser.parse_args()
    
    try:
        comprehensive_log, regular_log, max_iters_from_logs = find_log_files(args.directory)
        print(f"Analyzing log file: {comprehensive_log}")
        
        log_data = load_comprehensive_log(comprehensive_log)
        
        # Basic Diversity
        lexical = calculate_lexical_diversity(log_data)
        embedding = calculate_embedding_diversity(log_data)
        bleu = calculate_self_bleu(log_data)
        
        print(f"\nGlobal Diversity Metrics:")
        print(f"  Lexical Score: {lexical['diversity_score']:.3f}")
        print(f"  Embedding Div: {embedding:.3f}")
        print(f"  Self-BLEU:     {bleu:.3f}")
        
        # GA Specific Analysis
        ga_stats = analyze_ga_state(log_data)
            
        # Analysis
        analysis_results = analyze_rejection_patterns(log_data)
        category_analysis = analyze_by_category(log_data)
        
        # Print
        print_analysis(analysis_results, category_analysis, ga_stats)
        
        # Save
        output_path = args.output or os.path.join(args.directory, "ga_analysis_results.json")
        results = {
            'metrics': analysis_results,
            'ga_stats': ga_stats,
            'diversity': {'lexical': lexical, 'embedding': embedding, 'self_bleu': bleu},
            'category_analysis': category_analysis
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved analysis details to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()