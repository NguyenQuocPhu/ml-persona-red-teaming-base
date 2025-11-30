#!/usr/bin/env python3

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Analysis script for comprehensive logs from RainbowPlus.
Updated to include Attack Memory analysis.

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
    
    # Calculate success rate
    success_rate = accepted_count / total_prompts if total_prompts > 0 else 0
    
    # Analyze score distributions
    all_score_values = []
    all_similarity_values = []
    
    for key in all_scores:
        all_score_values.extend(all_scores[key])
    for key in all_similarities:
        all_similarity_values.extend(all_similarities[key])
    
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
        'success_rate': success_rate,
        'reason_counts': dict(reason_counts),
        'score_stats': get_stats(all_score_values),
        'similarity_stats': get_stats(all_similarity_values),
        'replaced_novelty': replaced_nov,
        'replaced_fitness': replaced_fit,
        'added_new_niche': added_new
    }


def analyze_by_category(log_data):
    """Analyze rejection patterns by category."""
    rejection_reasons = log_data.get('rejection_reasons', {})
    all_scores = log_data.get('all_scores', {})
    
    category_analysis = {}
    success_statuses = {'accepted', 'added_new_niche', 'replaced_novelty', 'replaced_fitness'}

    for key, reasons in rejection_reasons.items():
        try:
            category = eval(key)[0] if isinstance(key, str) else key[0]
        except:
            category = "unknown"
        
        if category not in category_analysis:
            category_analysis[category] = {
                'total': 0, 'accepted': 0, 
                'similarity_rejected': 0, 'fitness_rejected': 0,
                'scores': []
            }
        
        for reason in reasons:
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
    success_statuses = {'accepted', 'added_new_niche', 'replaced_novelty', 'replaced_fitness'}

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


def print_analysis(analysis_results, category_analysis, persona_analysis=None, mem_stats=None):
    """Print formatted analysis results."""
    print("=" * 60)
    print("COMPREHENSIVE LOG ANALYSIS")
    print("=" * 60)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total prompts generated: {analysis_results['total_prompts']}")
    print(f"Successfully accepted: {analysis_results['accepted_count']}")
    print(f"  - Added New Niche: {analysis_results.get('added_new_niche', 0)}")
    print(f"  - Replaced (Novelty): {analysis_results.get('replaced_novelty', 0)}")
    print(f"  - Replaced (Fitness): {analysis_results.get('replaced_fitness', 0)}")
    print(f"Rejected (Similarity): {analysis_results['similarity_rejected']}")
    print(f"Rejected (Fitness): {analysis_results['fitness_rejected']}")
    print(f"Success rate: {analysis_results['success_rate']:.2%}")
    
    print(f"\nSCORE STATISTICS:")
    s = analysis_results['score_stats']
    print(f"  Mean: {s['mean']:.3f}, Max: {s['max']:.3f}")

    if mem_stats:
        print(f"\nATTACK MEMORY STATISTICS:")
        print(f"  Total Saved: {mem_stats['total_entries']}")
        print(f"  Avg Score: {mem_stats['avg_score']:.3f}")
        print(f"  Max Score: {mem_stats['max_score']:.3f}")
        print("  Sample Entries:")
        for i, e in enumerate(mem_stats['sample_entries']):
             p = e.get('prompt', '')[:50].replace('\n', ' ')
             print(f"    {i+1}. [{e.get('score'):.2f}] {p}...")

    print(f"\nCATEGORY ANALYSIS:")
    for category, data in category_analysis.items():
        print(f"  {category:<20} | Total: {data['total']:<4} | Accepted: {data['accepted']:<4} ({data['success_rate']:.1%})")


def calculate_lexical_diversity(log_data):
    all_prompts = log_data.get('all_prompts', {})
    flat_prompts = []
    for key, prompts in all_prompts.items():
        flat_prompts.extend(prompts)
    
    if not flat_prompts: return {'diversity_score': 0}
    
    total = len(flat_prompts)
    unique = len(set(flat_prompts))
    return {'total_prompts': total, 'unique_prompts': unique, 'diversity_score': unique / total}


def calculate_embedding_diversity(log_data):
    all_prompts = log_data.get('all_prompts', {})
    flat_prompts = []
    for key, prompts in all_prompts.items():
        flat_prompts.extend(prompts)
    
    if len(flat_prompts) < 2: return 0.0
    
    # Sample for speed
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
    except Exception: return 0.0


def calculate_self_bleu(log_data, n=4):
    all_prompts = log_data.get('all_prompts', {})
    flat_prompts = []
    for key, prompts in all_prompts.items():
        flat_prompts.extend(prompts)
    
    if len(flat_prompts) < 2: return 0.0
    
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
    
    flat_prompts = []
    flat_reasons = []
    for key, prompts in all_prompts.items():
        flat_prompts.extend(prompts)
        flat_reasons.extend(rejection_reasons.get(key, [None]*len(prompts)))
    
    total_prompts = len(flat_prompts)
    
    # Filter pass rate
    num_pass_filter = sum(1 for r in flat_reasons if r != 'similarity_too_high')
    filter_pass_rate = num_pass_filter / total_prompts if total_prompts > 0 else 0

    # ASR (GA Success)
    ga_success = {'accepted', 'added_new_niche', 'replaced_novelty', 'replaced_fitness', 'replaced_fitness_improvement'}
    num_unsafe = sum(1 for r in flat_reasons if r in ga_success)
    ASR = num_unsafe / total_prompts if total_prompts > 0 else 0
    ASR_among_pass = num_unsafe / num_pass_filter if num_pass_filter else 0

    # Iteration ASR
    # (Simplified logic: assume uniform distribution if iteration data missing)
    iteration_ASR = 0 
    if num_iterations and num_iterations > 0:
        # Heuristic: did we find at least one success per 10 attempts?
        est_success_iters = min(num_iterations, num_unsafe) 
        iteration_ASR = est_success_iters / num_iterations

    return {
        'filter_pass_rate': filter_pass_rate,
        'ASR': ASR,
        'ASR_among_pass': ASR_among_pass,
        'iteration_ASR': iteration_ASR,
        'total_prompts': total_prompts,
        'num_unsafe': num_unsafe,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze comprehensive RainbowPlus logs")
    parser.add_argument("directory", help="Directory containing log files")
    parser.add_argument("--output", help="Output file for analysis results")
    parser.add_argument("--max-iterations", type=int)
    
    args = parser.parse_args()
    
    try:
        # 1. Find files
        comp_log, reg_log, mem_log, max_iters = find_log_files(args.directory)
        print(f"Comprehensive Log: {comp_log}")
        print(f"Regular Log: {reg_log}")
        print(f"Attack Memory Log: {mem_log}")
        
        if not comp_log:
            print("Error: No comprehensive log found.")
            return
            
        max_iterations = args.max_iterations or max_iters

        # 2. Load Data
        log_data = load_comprehensive_log(comp_log)
        memory_data = load_attack_memory_log(mem_log)
        
        # 3. Basic Metrics
        lexical_diversity = calculate_lexical_diversity(log_data)
        embedding_diversity = calculate_embedding_diversity(log_data)
        self_bleu = calculate_self_bleu(log_data)
        
        ga_metrics = calculate_ga_metrics(log_data)
        
        print("\nDIVERSITY SCORES:")
        print(f"  Lexical diversity: {lexical_diversity.get('diversity_score', 0):.3f}")
        print(f"  Embedding diversity: {embedding_diversity:.3f}")
        print(f"  Self-BLEU: {self_bleu:.3f}")
        
        if ga_metrics:
            print("\nGA ELITE METRICS:")
            print(f"  Elite Count: {ga_metrics['elite_count']}")
            print(f"  Elite Lexical Div: {ga_metrics['elite_lexical_diversity']:.3f}")

        # 4. Analysis
        metrics = calculate_comprehensive_metrics(log_data, max_iterations, regular_log)
        analysis_results = analyze_rejection_patterns(log_data)
        category_analysis = analyze_by_category(log_data)
        persona_analysis = analyze_by_persona(log_data)
        
        # Attack Memory Stats
        mem_stats = analyze_attack_memory(memory_data)

        # Print results
        print_analysis(analysis_results, category_analysis, persona_analysis, mem_stats)
        
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
            'ga_metrics': ga_metrics,
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