#!/usr/bin/env python3

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Analysis script for comprehensive logs from RainbowPlus.

This script helps analyze all generated prompts, including failed ones,
to understand rejection patterns and system effectiveness.

Usage:
    python analyze_comprehensive_logs.py <directory>
    python analyze_comprehensive_logs.py <directory> --max-iterations <num>
    
Examples:
    python analyze_comprehensive_logs.py logs-persona-fit/gpt-4o/harmbench
    python analyze_comprehensive_logs.py logs-persona-fit/gpt-4o/harmbench --max-iterations 30
    python analyze_comprehensive_logs.py logs-persona-fit/gpt-4o/harmbench --output custom_results.json
"""

import json
import argparse
import os
import glob
from pathlib import Path
from collections import Counter, defaultdict
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
    Automatically find comprehensive and regular log files in a directory.
    
    Args:
        directory: Path to the directory containing log files
        
    Returns:
        tuple: (comprehensive_log_path, regular_log_path, max_iters_from_logs)
    """
    directory = Path(directory)
    
    # Find comprehensive log file (prefer global, fallback to latest timestamped)
    comprehensive_global = directory / "comprehensive_ga_log_global.json"
    
    if comprehensive_global.exists():
        comprehensive_log = str(comprehensive_global)
    else:
        # Find timestamped comprehensive logs
        comprehensive_pattern = directory / "comprehensive_ga_log_*.json"
        comprehensive_candidates = list(glob.glob(str(comprehensive_pattern)))
        
        if not comprehensive_candidates:
            raise FileNotFoundError(f"No comprehensive log files found in {directory}")
        
        # Sort by timestamp and take the latest
        comprehensive_log = sorted(comprehensive_candidates)[-1]
    
    # Find regular log file (prefer global, fallback to latest timestamped)
    regular_log = None
    regular_global = directory / "ga_state_global.json"
    
    if regular_global.exists():
        regular_log = str(regular_global)
    else:
        # Find timestamped regular logs
        regular_pattern = directory / "ga_state_*.json"
        regular_candidates = list(glob.glob(str(regular_pattern)))
        
        if regular_candidates:
            # Sort by timestamp and take the latest
            regular_log = sorted(regular_candidates)[-1]
    
    # Extract max_iters from the log files (prefer comprehensive log)
    max_iters_from_logs = None
    
    # First try comprehensive log
    try:
        with open(comprehensive_log, 'r') as f:
            comp_data = json.load(f)
            max_iters_from_logs = comp_data.get('max_iters')
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    
    # If not found in comprehensive log, try regular log
    if max_iters_from_logs is None and regular_log:
        try:
            with open(regular_log, 'r') as f:
                reg_data = json.load(f)
                meta_data = reg_data.get('meta', {})
                max_iters_from_logs = meta_data.get('max_iters')
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
    
    return comprehensive_log, regular_log, max_iters_from_logs


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
    accepted_count = reason_counts.get('accepted', 0)
    similarity_rejected = reason_counts.get('similarity_too_high', 0)
    fitness_rejected = reason_counts.get('fitness_too_low', 0)
    replaced_novelty = reason_counts.get('replaced_novelty', 0)
    replaced_fitness = reason_counts.get('replaced_fitness', 0)

    # Calculate success rate
    success_rate = accepted_count / total_prompts if total_prompts > 0 else 0
    
    # Analyze score distributions
    all_score_values = []
    all_similarity_values = []
    
    for key in all_scores:
        all_score_values.extend(all_scores[key])
    for key in all_similarities:
        all_similarity_values.extend(all_similarities[key])
    
    score_stats = {
        'mean': np.mean(all_score_values) if all_score_values else 0,
        'std': np.std(all_score_values) if all_score_values else 0,
        'min': np.min(all_score_values) if all_score_values else 0,
        'max': np.max(all_score_values) if all_score_values else 0,
    }
    
    similarity_stats = {
        'mean': np.mean(all_similarity_values) if all_similarity_values else 0,
        'std': np.std(all_similarity_values) if all_similarity_values else 0,
        'min': np.min(all_similarity_values) if all_similarity_values else 0,
        'max': np.max(all_similarity_values) if all_similarity_values else 0,
    }
    
    return {
        'total_prompts': total_prompts,
        'accepted_count': accepted_count,
        'similarity_rejected': similarity_rejected,
        'fitness_rejected': fitness_rejected,
        'success_rate': success_rate,
        'reason_counts': dict(reason_counts),
        'score_stats': score_stats,
        'similarity_stats': similarity_stats,
        'replaced_novelty': replaced_novelty,
        'replaced_fitness': replaced_fitness,
    }


def analyze_by_category(log_data):
    """Analyze rejection patterns by category."""
    rejection_reasons = log_data.get('rejection_reasons', {})
    all_scores = log_data.get('all_scores', {})
    all_similarities = log_data.get('all_similarities', {})
    
    category_analysis = {}
    
    for key, reasons in rejection_reasons.items():
        # Parse the key to extract category information
        # Key format: ('Category', 'Style', 'Persona')
        try:
            category = eval(key)[0] if isinstance(key, str) else key[0]
        except:
            category = "unknown"
        
        if category not in category_analysis:
            category_analysis[category] = {
                'total': 0,
                'accepted': 0,
                'similarity_rejected': 0,
                'fitness_rejected': 0,
                'scores': [],
                'similarities': [],
            }
        
        # Count reasons for this category
        for reason in reasons:
            category_analysis[category]['total'] += 1
            if reason == 'accepted':
                category_analysis[category]['accepted'] += 1
            elif reason == 'similarity_too_high':
                category_analysis[category]['similarity_rejected'] += 1
            elif reason == 'fitness_too_low':
                category_analysis[category]['fitness_rejected'] += 1
            elif reason == 'replaced_novelty':
                category_analysis[category]['replaced_novelty'] += 1
            elif reason == 'replaced_fitness':
                category_analysis[category]['replaced_fitness'] += 1
        
        # Add scores and similarities
        if key in all_scores:
            category_analysis[category]['scores'].extend(all_scores[key])
        if key in all_similarities:
            category_analysis[category]['similarities'].extend(all_similarities[key])
    
    # Calculate success rates and statistics for each category
    for category, data in category_analysis.items():
        data['success_rate'] = data['accepted'] / data['total'] if data['total'] > 0 else 0
        data['avg_score'] = np.mean(data['scores']) if data['scores'] else 0
        data['avg_similarity'] = np.mean(data['similarities']) if data['similarities'] else 0
    
    return category_analysis


def analyze_by_persona(log_data):
    """Analyze rejection patterns by persona (third element of key tuple)."""
    rejection_reasons = log_data.get('rejection_reasons', {})
    all_scores = log_data.get('all_scores', {})
    all_similarities = log_data.get('all_similarities', {})
    
    persona_analysis = {}
    
    for key, reasons in rejection_reasons.items():
        # Key format: ('Category', 'Style', 'Persona')
        try:
            persona = eval(key)[2] if isinstance(key, str) else key[2]
        except:
            persona = "unknown"
        if persona not in persona_analysis:
            persona_analysis[persona] = {
                'total': 0,
                'accepted': 0,
                'similarity_rejected': 0,
                'fitness_rejected': 0,
                'scores': [],
                'similarities': [],
            }
        for reason in reasons:
            persona_analysis[persona]['total'] += 1
            if reason == 'accepted':
                persona_analysis[persona]['accepted'] += 1
            elif reason == 'similarity_too_high':
                persona_analysis[persona]['similarity_rejected'] += 1
            elif reason == 'fitness_too_low':
                persona_analysis[persona]['fitness_rejected'] += 1
            elif reason == 'replaced_novelty':
                persona_analysis[persona]['replaced_novelty'] += 1
            elif reason == 'replaced_fitness':
                persona_analysis[persona]['replaced_fitness'] += 1
        
        if key in all_scores:
            persona_analysis[persona]['scores'].extend(all_scores[key])
        if key in all_similarities:
            persona_analysis[persona]['similarities'].extend(all_similarities[key])
    for persona, data in persona_analysis.items():
        data['success_rate'] = data['accepted'] / data['total'] if data['total'] > 0 else 0
        data['avg_score'] = np.mean(data['scores']) if data['scores'] else 0
        data['avg_similarity'] = np.mean(data['similarities']) if data['similarities'] else 0
    return persona_analysis


def print_analysis(analysis_results, category_analysis, persona_analysis=None):
    """Print formatted analysis results."""
    print("=" * 60)
    print("COMPREHENSIVE LOG ANALYSIS")
    print("=" * 60)
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"Total prompts generated: {analysis_results['total_prompts']}")
    print(f"Successfully accepted: {analysis_results['accepted_count']}")
    print(f"Rejected for similarity: {analysis_results['similarity_rejected']}")
    print(f"Rejected for fitness: {analysis_results['fitness_rejected']}")
    print(f"Replaced for novelty: {analysis_results['replaced_novelty']}")
    print(f"Replaced for fitness: {analysis_results['replaced_fitness']}")
    print(f"Success rate: {analysis_results['success_rate']:.2%}")
    
    print(f"\nREJECTION REASON BREAKDOWN:")
    for reason, count in analysis_results['reason_counts'].items():
        percentage = count / analysis_results['total_prompts'] * 100
        print(f"  {reason}: {count} ({percentage:.1f}%)")
    
    print(f"\nSCORE STATISTICS:")
    score_stats = analysis_results['score_stats']
    print(f"  Mean: {score_stats['mean']:.3f}")
    print(f"  Std: {score_stats['std']:.3f}")
    print(f"  Range: [{score_stats['min']:.3f}, {score_stats['max']:.3f}]")
    
    print(f"\nSIMILARITY STATISTICS:")
    sim_stats = analysis_results['similarity_stats']
    print(f"  Mean: {sim_stats['mean']:.3f}")
    print(f"  Std: {sim_stats['std']:.3f}")
    print(f"  Range: [{sim_stats['min']:.3f}, {sim_stats['max']:.3f}]")
    
    print(f"\nCATEGORY ANALYSIS:")
    for category, data in category_analysis.items():
        print(f"\n  {category}:")
        print(f"    Total: {data['total']}")
        print(f"    Accepted: {data['accepted']} ({data['success_rate']:.2%})")
        print(f"    Avg Score: {data['avg_score']:.3f}")
        print(f"    Avg Similarity: {data['avg_similarity']:.3f}")
    if persona_analysis:
        print(f"\nPERSONA ANALYSIS:")
        for persona, data in persona_analysis.items():
            print(f"  {persona}:")
            print(f"    Total: {data['total']}")
            print(f"    Accepted: {data['accepted']}")
            print(f"    Success rate: {data['success_rate']:.2%}")
            print(f"    Avg Score: {data['avg_score']:.3f}")
            print(f"    Avg Similarity: {data['avg_similarity']:.3f}")


def calculate_lexical_diversity(log_data):
    # Gather all mutated prompts
    all_prompts = log_data.get('all_prompts', {})
    flat_prompts = []
    for key, prompts in all_prompts.items():
        flat_prompts.extend(prompts)
    total = len(flat_prompts)
    unique = len(set(flat_prompts))
    diversity_score = unique / total if total > 0 else 0
    return {
        'total_prompts': total,
        'unique_prompts': unique,
        'diversity_score': diversity_score
    }


def calculate_embedding_diversity(log_data):
    """Calculate embedding-based diversity from all prompts."""
    all_prompts = log_data.get('all_prompts', {})
    flat_prompts = []
    for key, prompts in all_prompts.items():
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


def calculate_self_bleu(log_data, n=4):
    """
    Calculate Self-BLEU for the set of prompts.
    Lower is more diverse.
    """
    all_prompts = log_data.get('all_prompts', {})
    flat_prompts = []
    for key, prompts in all_prompts.items():
        flat_prompts.extend(prompts)
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

def calculate_lexical_ga_diversity(log_data):
    # Gather all mutated prompts
    ga_state = log_data.get('ga_state', {})
    elites = ga_state.get('elites', {})
    flat_prompts = [
        elite_data.get('prompt')
        for elite_data in elites.values()
        if elite_data.get('prompt') is not None
    ]
    total = len(flat_prompts)
    unique = len(set(flat_prompts))
    diversity_score = unique / total if total > 0 else 0
    return {
        'total_prompts': total,
        'unique_prompts': unique,
        'diversity_score': diversity_score
    }


def calculate_embedding_ga_diversity(log_data):
    """Calculate embedding-based diversity from all prompts."""
    ga_state = log_data.get('ga_state', {})
    elites = ga_state.get('elites', {})
    flat_prompts = [
        elite_data.get('prompt')
        for elite_data in elites.values()
        if elite_data.get('prompt') is not None
    ]
    
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


def calculate_self_ga_bleu(log_data, n=4):
    """
    Calculate Self-BLEU for the set of prompts.
    Lower is more diverse.
    """
    ga_state = log_data.get('ga_state', {})
    elites = ga_state.get('elites', {})
    flat_prompts = [
        elite_data.get('prompt')
        for elite_data in elites.values()
        if elite_data.get('prompt') is not None
    ]
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


def calculate_comprehensive_metrics(log_data, num_iterations=None, regular_log_path=None):
    """
    Calculate:
    1. filter_pass_rate: Out of all prompts, how many passed the similarity filter
    2. ASR: Out of all prompts, how many were classified as unsafe
    3. ASR_among_pass: Out of prompts that passed the filter, how many were unsafe
    4. iteration_ASR: Out of all iterations, how many produced at least one unsafe response
    """
    all_prompts = log_data.get('all_prompts', {})
    rejection_reasons = log_data.get('rejection_reasons', {})
    all_scores = log_data.get('all_scores', {})
    all_similarities = log_data.get('all_similarities', {})

    # Flatten all prompts and rejection reasons
    flat_prompts = []
    flat_reasons = []
    flat_keys = []
    flat_scores = []
    for key, scores in all_scores.items():
        flat_scores.extend(scores)
    for key, prompts in all_prompts.items():
        flat_prompts.extend(prompts)
        flat_reasons.extend(rejection_reasons.get(key, [None]*len(prompts)))
        flat_keys.extend([key]*len(prompts))
    total_prompts = len(flat_prompts)

    # 1. filter_pass_rate
    num_pass_filter = sum(1 for r in flat_reasons if r != 'similarity_too_high')
    filter_pass_rate = num_pass_filter / total_prompts if total_prompts > 0 else 0

    # 2. ASR (Attack Success Rate): classified as unsafe out of all prompts
    num_unsafe = sum(1 for r in flat_reasons if r == 'accepted')
    ASR = num_unsafe / total_prompts if total_prompts > 0 else 0

    # 3. ASR_among_pass: unsafe out of those that passed the filter
    ASR_among_pass = num_unsafe / num_pass_filter if num_pass_filter else 0

    # 4. iteration_ASR: Out of all iterations, how many produced at least one unsafe response
    if num_iterations is not None and regular_log_path is not None:
        try:
            with open(regular_log_path, 'r') as f:
                regular_log = json.load(f)
            iters_data = regular_log.get('iters', {})
            iterations_with_success = set()
            for key, iterations in iters_data.items():
                # Only count integer iteration numbers
                iterations_with_success.update(i for i in iterations if isinstance(i, int))
            num_iterations_total = num_iterations
            num_iterations_with_unsafe = len(iterations_with_success)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            num_iterations_total = num_iterations
            num_iterations_with_unsafe = 1 if num_unsafe > 0 else 0
    elif num_iterations is not None:
        num_iterations_total = num_iterations
        num_iterations_with_unsafe = 1 if num_unsafe > 0 else 0
    else:
        num_iterations_total = 1
        num_iterations_with_unsafe = 1 if num_unsafe > 0 else 0
    
    iteration_ASR = num_iterations_with_unsafe / num_iterations_total if num_iterations_total > 0 else 0
    # 5. Avg fitness: Average fitness of all succesful prompts
    fitness_accepted = [r for r in flat_reasons if r == 'accepted']
    avg_fitness = np.mean(fitness_accepted)
    # 6. Top k prompts: adversarial with my methods
    k = min(k, len(flat_prompts))
    scored_prompts = list(zip(flat_prompts, flat_scores))
    scored_prompts.sort(key=lambda x: x[1], reverse=True)
    top_k_prompts = scored_prompts[:k]

    return {
        'filter_pass_rate': filter_pass_rate,
        'ASR': ASR,
        'ASR_among_pass': ASR_among_pass,
        'iteration_ASR': iteration_ASR,
        'num_iterations': num_iterations_total,
        'num_iterations_with_unsafe': num_iterations_with_unsafe,
        'total_prompts': total_prompts,
        'num_pass_filter': num_pass_filter,
        'num_unsafe': num_unsafe,
        'avg_fitness': avg_fitness 
    }
def calculate_ga_comprehensive_metrics(log_data, num_iterations=None, regular_log_path=None):
    """
    Calculate:
    1. filter_pass_rate: Out of all prompts, how many passed the similarity filter
    2. ASR_among_pass: Out of prompts that passed the filter, how many were unsafe

    3. ASR: Out of all prompts, how many were classified as unsafe
    4. iteration_ASR: Out of all iterations, how many produced at least one unsafe response
    """
    all_prompts = log_data.get('all_prompts', {})
    rejection_reasons = log_data.get('rejection_reasons', {})
    all_scores = log_data.get('all_scores', {})
    all_similarities = log_data.get('all_similarities', {})

    # Flatten all prompts and rejection reasons
    flat_prompts = []
    flat_reasons = []
    flat_keys = []
    for key, prompts in all_prompts.items():
        flat_prompts.extend(prompts)
        flat_reasons.extend(rejection_reasons.get(key, [None]*len(prompts)))
        flat_keys.extend([key]*len(prompts))
    total_prompts = len(flat_prompts)

    # 1. filter_pass_rate
    num_pass_filter = sum(1 for r in flat_reasons if r != 'similarity_too_high')
    filter_pass_rate = num_pass_filter / total_prompts if total_prompts > 0 else 0

    # 2. ASR (Attack Success Rate): classified as unsafe out of all prompts
    num_unsafe = sum(1 for r in flat_reasons if r == 'accepted')
    ASR = num_unsafe / total_prompts if total_prompts > 0 else 0

    # 3. ASR_among_pass: unsafe out of those that passed the filter
    ASR_among_pass = num_unsafe / num_pass_filter if num_pass_filter else 0

    # 4. iteration_ASR: Out of all iterations, how many produced at least one unsafe response
    if num_iterations is not None and regular_log_path is not None:
        try:
            with open(regular_log_path, 'r') as f:
                regular_log = json.load(f)
            iters_data = regular_log.get('iters', {})
            iterations_with_success = set()
            for key, iterations in iters_data.items():
                # Only count integer iteration numbers
                iterations_with_success.update(i for i in iterations if isinstance(i, int))
            num_iterations_total = num_iterations
            num_iterations_with_unsafe = len(iterations_with_success)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            num_iterations_total = num_iterations
            num_iterations_with_unsafe = 1 if num_unsafe > 0 else 0
    elif num_iterations is not None:
        num_iterations_total = num_iterations
        num_iterations_with_unsafe = 1 if num_unsafe > 0 else 0
    else:
        num_iterations_total = 1
        num_iterations_with_unsafe = 1 if num_unsafe > 0 else 0
    
    iteration_ASR = num_iterations_with_unsafe / num_iterations_total if num_iterations_total > 0 else 0

    ga_state = log_data.get('ga_state', {})
    if ga_state:
        success_statuses = {
            'accepted', 'added_new_niche', 'replaced_novelty', 
            'replaced_fitness', 'replaced_fitness_improvement',
            'rejected_not_elite' # <--- QUAN TRỌNG
        }
        # 5. ASR (Attack Success Rate): classified as unsafe out of all prompts used to GA
        num_iterations_with_unsafe_ga = sum(1 for r in flat_reasons if r in success_statuses)
        ASR_GA = num_iterations_with_unsafe_ga / total_prompts if total_prompts > 0 else 0

        # 2.Avg fitness: average fitness of all unsafe prompts
        elites = ga_state.get('elites', {})
        fitness_value = []
        for elite_data in elites.values():
            if 'fitness' in elite_data:
                fitness_value.append(elite_data['fitness'])
        avg_fitness = np.mean(fitness_value) if fitness_value else 0

    
    return {
        'filter_pass_rate': filter_pass_rate,
        'ASR': ASR,
        'ASR_among_pass': ASR_among_pass,
        'iteration_ASR': iteration_ASR,
        'num_iterations': num_iterations_total,
        'num_iterations_with_unsafe': num_iterations_with_unsafe,
        'total_prompts': total_prompts,
        'num_pass_filter': num_pass_filter,
        'num_unsafe': num_unsafe,
        'num_iterations_with_unsafe_ga': num_iterations_with_unsafe_ga if num_iterations_with_unsafe_ga else 0,
        'ASR_GA': ASR_GA if ASR_GA else 0,
        'avg_fitness': avg_fitness if avg_fitness else 0,
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze comprehensive RainbowPlus logs")
    parser.add_argument("directory", help="Directory containing log files")
    parser.add_argument("--output", help="Output file for analysis results (JSON, defaults to analysis_results.json in input directory)")
    parser.add_argument("--max-iterations", type=int, help="Maximum iterations used in the experiment (for accurate iteration_ASR calculation)")
    
    args = parser.parse_args()
    
    try:
        # Automatically find log files
        comprehensive_log, regular_log, max_iters_from_logs = find_log_files(args.directory)
        print(f"use comprehensive log: {comprehensive_log}")
        print(f"use regular log: {regular_log}")
        
        # Determine max_iterations: use command line arg, then log file, then None
        max_iterations = getattr(args, 'max_iterations', None) or max_iters_from_logs
        
        if max_iters_from_logs:
            print(f"max_iters: {max_iters_from_logs}")
        else:
            print("No max_iters found in log files (older logs generated before max_iters was saved)")
            
        if max_iterations:
            print(f"Using {max_iterations} iterations for analysis")
        else:
            print("Warning: No iteration count available - iteration_ASR may be inaccurate")
            print("Consider providing --max-iterations parameter for older logs")
        
        # Set default output path if not specified
        output_path = args.output
        if output_path is None:
            output_path = os.path.join(args.directory, "analysis_results.json")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error processing directory: {e}")
        return
    
    # Load log data
    log_data = load_comprehensive_log(comprehensive_log)
    
    # Diversity scores
    lexical_diversity = calculate_lexical_diversity(log_data)
    embedding_diversity = calculate_embedding_diversity(log_data)
    self_bleu = calculate_self_bleu(log_data)
    
    lexical_ga_diversity = calculate_lexical_ga_diversity(log_data)
    embedding_ga_diversity = calculate_embedding_ga_diversity(log_data)
    self_bleu_ga = calculate_self_ga_bleu(log_data)
    
    print("\nDIVERSITY SCORES:")
    print(f"  Lexical diversity:")
    print(f"    Unique prompts: {lexical_diversity['unique_prompts']}")
    print(f"    Total prompts: {lexical_diversity['total_prompts']}")
    print(f"    Diversity score (unique/total): {lexical_diversity['diversity_score']:.3f}")
    print(f"  Embedding diversity (avg pairwise cosine distance): {embedding_diversity:.4f}")
    print(f"  Self-BLEU (lower is more diverse): {self_bleu:.4f}")

    print("\nGA DIVERSITY SCORES:")
    print(f"  Lexical diversity:")
    print(f"    Unique prompts: {lexical_ga_diversity['unique_prompts']}")
    print(f"    Total prompts: {lexical_ga_diversity['total_prompts']}")
    print(f"    Diversity score (unique/total): {lexical_ga_diversity['diversity_score']:.3f}")
    print(f"  Embedding diversity (avg pairwise cosine distance): {embedding_ga_diversity:.4f}")
    print(f"  Self-BLEU (lower is more diverse): {self_bleu_ga:.4f}")

    # Comprehensive metrics using automatically extracted parameters
    metrics = calculate_comprehensive_metrics(log_data, max_iterations, regular_log)
    print("\nCOMPREHENSIVE METRICS:")
    print(f"  filter_pass_rate: {metrics['filter_pass_rate']:.3f}")
    print(f"  ASR: {metrics['ASR']:.3f}")
    print(f"  ASR_among_pass: {metrics['ASR_among_pass']:.3f}")
    print(f"  iteration_ASR: {metrics['iteration_ASR']:.3f} ({metrics['num_iterations_with_unsafe']}/{metrics['num_iterations']})")
    print(f"  Total prompts: {metrics['total_prompts']}")
    print(f"  Prompts passed filter: {metrics['num_pass_filter']}")
    print(f"  Unsafe prompts: {metrics['num_unsafe']}")

    # Analyze patterns
    analysis_results = analyze_rejection_patterns(log_data)
    category_analysis = analyze_by_category(log_data)
    persona_analysis = analyze_by_persona(log_data)
    
    # Print results
    print_analysis(analysis_results, category_analysis, persona_analysis)
    
    # Save results to output file
    results = {
        'log_directory': args.directory,
        'comprehensive_log_file': comprehensive_log,
        'regular_log_file': regular_log,
        'max_iterations': max_iterations,
        'comprehensive_metrics': metrics,
        'diversity': {
            'lexical': lexical_diversity,
            'embedding': embedding_diversity,
            'self_bleu': self_bleu,
        },
        'overall_analysis': analysis_results,
        'category_analysis': category_analysis,
        'persona_analysis': persona_analysis,
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis results saved to {output_path}")


if __name__ == "__main__":
    main() 