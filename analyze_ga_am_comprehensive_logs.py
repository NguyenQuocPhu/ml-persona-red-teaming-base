#!/usr/bin/env python3

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Analysis script for comprehensive GA logs from RainbowPlus.
Updated to include Attack Memory analysis and fix format string errors.

Usage:
    python analyze_comprehensive_logs.py <directory>
"""

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
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load attack memory log: {e}")
        return None

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

    # Extract max_iters
    max_iters_from_logs = None
    if comprehensive_log:
        try:
            with open(comprehensive_log, 'r') as f:
                comp_data = json.load(f)
                if 'meta' in comp_data: # New format
                    max_iters_from_logs = comp_data['meta'].get('max_iters')
                else: # Old format
                    max_iters_from_logs = comp_data.get('max_iters')
        except Exception: pass
    
    return comprehensive_log, regular_log, attack_memory_log, max_iters_from_logs


def analyze_rejection_patterns(log_data):
    """Analyze rejection patterns in the comprehensive log."""
    rejection_reasons = log_data.get('rejection_reasons', {})
    all_scores = log_data.get('all_scores', {})
    all_similarities = log_data.get('all_similarities', {})
    
    # Flatten all rejection reasons
    all_reasons = []
    for key, reasons in rejection_reasons.items():
        all_reasons.extend(reasons)
    
    # Count rejection reasons
    reason_counts = Counter(all_reasons)
    
    # Calculate statistics
    total_prompts = len(all_reasons)
    
    # Accepted counts (Include GA statuses)
    accepted_base = reason_counts.get('accepted', 0)
    added_new = reason_counts.get('added_new_niche', 0)
    replaced_nov = reason_counts.get('replaced_novelty', 0)
    replaced_fit = reason_counts.get('replaced_fitness', 0)
    
    accepted_count = accepted_base + added_new + replaced_nov + replaced_fit
    
    similarity_rejected = reason_counts.get('similarity_too_high', 0)
    fitness_rejected = reason_counts.get('fitness_too_low', 0) + reason_counts.get('rejected_fitness_low', 0)
    
    # GA specific rejection (Passed hard filters but failed elite competition)
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
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }

    return {
        'total_prompts': total_prompts,
        'accepted_count': accepted_count,
        'similarity_rejected': similarity_rejected,
        'fitness_rejected': fitness_rejected,
        'ga_rejected': ga_rejected, # New metric for GA density rejection
        'success_rate': success_rate,
        'reason_counts': dict(reason_counts),
        'score_stats': get_stats(all_score_values),
        'similarity_stats': get_stats(all_similarity_values),
        'replaced_novelty': replaced_nov,
        'replaced_fitness': replaced_fit,
        'added_new_niche': added_new
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
    
    # Define success statuses
    success_statuses = {'accepted', 'added_new_niche', 'replaced_novelty', 'replaced_fitness', 'replaced_fitness_improvement'}
    
    for key, reasons in rejection_reasons.items():
        try:
            # Handle tuple string representation
            category = eval(key)[0] if isinstance(key, str) else key[0]
        except:
            category = "unknown"
        
        if category not in category_analysis:
            category_analysis[category] = {
                'total': 0, 'accepted': 0, 
                'similarity_rejected': 0, 'fitness_rejected': 0,
                'scores': []
            }
        
        for i, reason in enumerate(reasons):
            category_analysis[category]['total'] += 1
            if reason in success_statuses:
                category_analysis[category]['accepted'] += 1
            elif reason == 'similarity_too_high':
                category_analysis[category]['similarity_rejected'] += 1
            elif reason in ['fitness_too_low', 'rejected_fitness_low']:
                category_analysis[category]['fitness_rejected'] += 1
        
        if key in all_scores:
            category_analysis[category]['scores'].extend(all_scores[key])
    
    for category, data in category_analysis.items():
        data['success_rate'] = data['accepted'] / data['total'] if data['total'] > 0 else 0
        scores = data.pop('scores') # Remove list to save space
        data['avg_score'] = float(np.mean(scores)) if scores else 0.0
    
    return category_analysis


def analyze_by_persona(log_data):
    """Analyze rejection patterns by persona."""
    rejection_reasons = log_data.get('rejection_reasons', {})
    all_scores = log_data.get('all_scores', {})
    
    persona_analysis = {}
    success_statuses = {'accepted', 'added_new_niche', 'replaced_novelty', 'replaced_fitness', 'replaced_fitness_improvement'}

    for key, reasons in rejection_reasons.items():
        try:
            persona = eval(key)[2] if isinstance(key, str) else key[2]
        except:
            persona = "unknown"
            
        if persona not in persona_analysis:
            persona_analysis[persona] = {'total': 0, 'accepted': 0, 'scores': []}
            
        for reason in reasons:
            persona_analysis[persona]['total'] += 1
            if reason in success_statuses:
                persona_analysis[persona]['accepted'] += 1
        
        if key in all_scores:
            persona_analysis[persona]['scores'].extend(all_scores[key])

    for persona, data in persona_analysis.items():
        data['success_rate'] = data['accepted'] / data['total'] if data['total'] > 0 else 0
        scores = data.pop('scores')
        data['avg_score'] = float(np.mean(scores)) if scores else 0.0
        
    return persona_analysis


def analyze_attack_memory(memory_data):
    """Analyze the loaded attack memory."""
    if not memory_data:
        return None
        
    total_entries = len(memory_data)
    scores = [entry.get('score', 0) for entry in memory_data]
    
    avg_score = np.mean(scores) if scores else 0
    max_score = np.max(scores) if scores else 0
    min_score = np.min(scores) if scores else 0
    
    return {
        'total_entries': total_entries,
        'avg_score': avg_score,
        'max_score': max_score,
        'min_score': min_score,
        'sample_entries': memory_data[:3]
    }

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
    elif 'temp' in log_data:
        flat_prompts = log_data['temp']
    else:
        return 0.0
    
    if len(flat_prompts) < 2:
        return 0.0
    
    # Optimization: Limit samples for speed in analysis script
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
    elif 'temp' in log_data:
        flat_prompts = log_data['temp']
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

def calculate_ga_metrics(log_data):
    """Calculate metrics specifically for the Growing Archive state."""
    ga_state = log_data.get('ga_state', {})
    elites = ga_state.get('elites', {})
    
    elite_prompts = []
    for e in elites.values():
        if 'prompt' in e: elite_prompts.append(e['prompt'])
        
    if not elite_prompts: return {}
    
    # Calculate diversity of elites
    unique = len(set(elite_prompts))
    lex_div = unique / len(elite_prompts) if elite_prompts else 0
    
    return {
        'n_centroids': ga_state.get('n_centroids', 0),
        'elite_lexical_diversity': lex_div,
        'elite_count': len(elite_prompts)
    }

def calculate_comprehensive_metrics(log_data, num_iterations=None, regular_log_path=None):
    all_prompts = log_data.get('all_prompts', {})
    rejection_reasons = log_data.get('rejection_reasons', {})
    all_iterations = log_data.get('all_iterations', {}) 
    
    flat_reasons = []
    flat_iterations = []
    
    # Flatten and align data
    for key in all_prompts:
        count = len(all_prompts[key])
        reasons = rejection_reasons.get(key, [])
        iters = all_iterations.get(key, [])
        
        # Pad if necessary
        flat_reasons.extend(reasons[:count] + ['unknown']*(count-len(reasons)))
        flat_iterations.extend(iters[:count] + [-1]*(count-len(iters)))

    total_prompts = len(flat_reasons)
    
    # Success statuses
    ga_success = {
        'accepted', 'added_new_niche', 'replaced_novelty', 
        'replaced_fitness', 'replaced_fitness_improvement', 'rejected_not_elite'
    }
    
    num_unsafe = sum(1 for r in flat_reasons if r in ga_success)
    
    # ASR metrics
    ASR = num_unsafe / total_prompts if total_prompts > 0 else 0
    
    # Filter Pass Rate
    hard_filters = ['similarity_too_high', 'fitness_too_low', 'rejected_fitness_low']
    num_pass_filter = sum(1 for r in flat_reasons if r not in hard_filters)
    filter_pass_rate = num_pass_filter / total_prompts if total_prompts > 0 else 0
    ASR_among_pass = num_unsafe / num_pass_filter if num_pass_filter else 0

    # Iteration ASR Calculation
    # Find iterations with at least one success
    successful_iters = set()
    valid_iters = set()
    
    # Try to get max iteration count from different sources
    if num_iterations:
        total_iters = num_iterations
    else:
        # Infer from log data if possible, else default to 1
        max_iter_in_log = max([i for i in flat_iterations if isinstance(i, int) and i >= 0], default=-1)
        total_iters = max(1, max_iter_in_log + 1)

    for r, it in zip(flat_reasons, flat_iterations):
        if isinstance(it, int) and it >= 0:
            valid_iters.add(it)
            if r in ga_success:
                successful_iters.add(it)
    
    iteration_ASR = len(successful_iters) / total_iters if total_iters > 0 else 0

    return {
        'filter_pass_rate': filter_pass_rate,
        'ASR': ASR,
        'ASR_among_pass': ASR_among_pass,
        'iteration_ASR': iteration_ASR,
        'total_prompts': total_prompts,
        'num_unsafe': num_unsafe,
        'successful_iterations': len(successful_iters),
        'total_iterations': total_iters
    }

def print_analysis(analysis_results, category_analysis, persona_analysis=None, mem_stats=None, ga_stats=None, metrics=None):
    """Print formatted analysis results."""
    print("=" * 60)
    print("COMPREHENSIVE LOG ANALYSIS")
    print("=" * 60)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total Prompts Generated: {analysis_results['total_prompts']}")
    print(f"Successful Attacks (Cumulative): {analysis_results['accepted_count']}")
    if metrics:
        print(f"ASR (Attack Success Rate): {metrics['ASR']:.2%}")
        print(f"Iteration ASR: {metrics['iteration_ASR']:.2%} ({metrics['successful_iterations']}/{metrics['total_iterations']} iters)")
    
    print(f"\nGA OPERATIONS:")
    print(f"  - Added New Niche: {analysis_results.get('added_new_niche', 0)}")
    print(f"  - Replaced (Novelty): {analysis_results.get('replaced_novelty', 0)}")
    print(f"  - Replaced (Fitness): {analysis_results.get('replaced_fitness', 0)}")
    
    print(f"\nREJECTIONS:")
    print(f"  - Similarity Too High: {analysis_results['similarity_rejected']}")
    print(f"  - Fitness Too Low: {analysis_results['fitness_rejected']}")
    print(f"  - Not Elite (GA Density): {analysis_results.get('ga_rejected', 0)}")
    
    print(f"\nSCORE STATISTICS:")
    s = analysis_results['score_stats']
    print(f"  Mean: {s['mean']:.3f}, Max: {s['max']:.3f}")

    if ga_stats:
        print(f"\nGROWING ARCHIVE STATE:")
        print(f"  Active Centroids: {ga_stats['n_centroids']}")
        
        # --- SỬA LỖI FORMAT Ở ĐÂY ---
        # Kiểm tra kiểu dữ liệu trước khi format
        dmin_val = ga_stats['dmin']
        if isinstance(dmin_val, (int, float)):
             print(f"  Threshold (dmin): {dmin_val:.4f}")
        else:
             print(f"  Threshold (dmin): {dmin_val}")
             
        print(f"  Survivor Diversity: {ga_stats['elite_diversity']:.3f}")

    if mem_stats:
        print(f"\nATTACK MEMORY STATISTICS:")
        print(f"  Total Saved: {mem_stats['total_entries']}")
        print(f"  Avg Score: {mem_stats['avg_score']:.2f}")
        print("  Sample Entries:")
        for i, e in enumerate(mem_stats['sample_entries']):
             p = e.get('prompt', '')[:50].replace('\n', ' ')
             print(f"    {i+1}. [{e.get('score'):.2f}] {p}...")

    print(f"\nCATEGORY ANALYSIS:")
    for category, data in category_analysis.items():
        print(f"  {category:<20} | Total: {data['total']:<4} | Accepted: {data['accepted']:<4} ({data['success_rate']:.1%})")

def main():
    parser = argparse.ArgumentParser(description="Analyze comprehensive RainbowPlus logs")
    parser.add_argument("directory", help="Directory containing log files")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--max-iterations", type=int)
    
    args = parser.parse_args()
    
    try:
        # 1. Find files
        # Note the variable name consistency here!
        comprehensive_log, regular_log, attack_memory_log, max_iters_from_logs = find_log_files(args.directory)
        
        print(f"Comprehensive Log: {comprehensive_log}")
        print(f"Regular Log: {regular_log}")
        print(f"Attack Memory Log: {attack_memory_log}")
        
        if not comprehensive_log:
            print("Error: No comprehensive log found.")
            return

        # 2. Load Data
        log_data = load_comprehensive_log(comprehensive_log)
        memory_data = load_attack_memory_log(attack_memory_log)
        
        # Determine max_iterations
        max_iterations = args.max_iterations or max_iters_from_logs
        
        # 3. Basic Diversity
        lexical_diversity = calculate_lexical_diversity(log_data)
        embedding_diversity = calculate_embedding_diversity(log_data)
        self_bleu = calculate_self_bleu(log_data)
        
        print("\nDIVERSITY SCORES:")
        print(f"  Lexical diversity: {lexical_diversity['diversity_score']:.3f}")
        print(f"  Embedding diversity: {embedding_diversity:.3f}")
        print(f"  Self-BLEU: {self_bleu:.3f}")
        
        # GA Specific Analysis
        ga_stats = analyze_ga_state(log_data)
        if ga_stats:
            print(f"\nGA ELITE METRICS:")
            print(f"  Elite Count: {ga_stats['n_elites']}")
            print(f"  Elite Lexical Div: {ga_stats['elite_diversity']:.3f}")

        # 4. Analysis
        # --- FIXED CALL HERE: using regular_log variable correctly ---
        metrics = calculate_comprehensive_metrics(log_data, max_iterations, regular_log)
        
        analysis_results = analyze_rejection_patterns(log_data)
        category_analysis = analyze_by_category(log_data)
        persona_analysis = analyze_by_persona(log_data)
        
        # Attack Memory Stats
        mem_stats = analyze_attack_memory(memory_data)

        # Print results
        print_analysis(analysis_results, category_analysis, persona_analysis, mem_stats, ga_stats, metrics)
        
        # Save results
        output_path = args.output or os.path.join(args.directory, "analysis_results.json")
        results = {
            'log_directory': args.directory,
            'comprehensive_metrics': metrics,
            'diversity': {
                'lexical': lexical_diversity,
                'embedding': embedding_diversity,
                'self_bleu': self_bleu,
            },
            'ga_metrics': ga_stats,
            'attack_memory_stats': mem_stats,
            'overall_analysis': analysis_results,
            'category_analysis': category_analysis,
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nAnalysis results saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()