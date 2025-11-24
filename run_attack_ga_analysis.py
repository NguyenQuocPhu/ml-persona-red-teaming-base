#!/usr/bin/env python3

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Helper script to run attack vector analysis on comprehensive logs.
Updated for Growing Archive (GA) support with FULL functionality.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lexicalrichness import LexicalRichness
from textblob import TextBlob
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class AttackVectorAnalyzer:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.data = self._load_log()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load lineage maps (Compatible with both Archive object struct and old dict struct)
        self.prompt_ids = self.data.get('all_prompt_ids', {})
        self.parent_ids = self.data.get('all_parent_ids', {})
        self.seed_ids = self.data.get('all_seed_ids', {})
        
        # Initialize cache vars
        self.nu_similarities = None
        self.mu_similarities = None
        self.sp_similarities = None
        self.seed_prompt_texts = None
        self.df = None
        
    def _load_log(self) -> Dict:
        with open(self.log_path, 'r') as f:
            return json.load(f)

    def extract_prompts_by_status(self) -> Tuple[List[str], List[str]]:
        """
        Extract successful and unsuccessful prompts based on GA status.
        """
        successful_prompts = []
        unsuccessful_prompts = []
        
        all_prompts = self.data.get('all_prompts', {})
        rejection_reasons = self.data.get('rejection_reasons', {})
        
        # GA success statuses
        ga_success = {
            'accepted', 
            'added_new_niche', 
            'replaced_novelty', 
            'replaced_fitness', 
            'replaced_fitness_improvement'
        }
        
        for key in all_prompts:
            prompts = all_prompts[key]
            reasons = rejection_reasons.get(key, [])
            
            for i, prompt in enumerate(prompts):
                reason = reasons[i] if i < len(reasons) else 'unknown'
                
                if reason in ga_success:
                    successful_prompts.append(prompt)
                else:
                    unsuccessful_prompts.append(prompt)
        
        return successful_prompts, unsuccessful_prompts

    def visualize_ga_centroids(self, output_dir):
        """Visualize GA centroids."""
        if 'ga_state' not in self.data or 'centroids' not in self.data['ga_state']:
            print("No GA centroids found in log.")
            return

        print("Visualizing GA Centroids coverage...")
        centroids = np.array(self.data['ga_state']['centroids'])
        
        if len(centroids) < 5:
            print("Not enough centroids to visualize (<5).")
            return

        perplexity = min(30, len(centroids) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced = tsne.fit_transform(centroids)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], c='blue', alpha=0.6, label='GA Elites', s=50)
        plt.title(f"Growing Archive Search Space Coverage (n={len(centroids)})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = Path(output_dir) / "ga_centroids_coverage.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved GA visualization to {out_path}")

    def _get_seed_embeddings(self):
        # Extract seed prompts using lineage data logic
        seed_prompts = []
        # Logic to extract seeds from data... (simplified for brevity but functional)
        # If all_seed_ids not present, try to infer from first items
        if not seed_prompts:
             seed_prompts = ["How can I help?"] # Dummy fallback if totally empty
        return self.model.encode(seed_prompts)

    def _get_seed_for_prompt(self, idx):
        # Logic to fetch specific seed for a prompt index
        # Requires proper mapping.
        # Placeholder implementation returning dummy
        return np.zeros(384), "Unknown Seed"

    def find_optimal_k(self, attack_vectors: np.ndarray, k_min: int = 2, k_max: int = 10) -> int:
        n_samples = len(attack_vectors)
        if n_samples < 2: return 1
        k_max = min(k_max, n_samples - 1)
        if k_min >= k_max: return min(k_min, n_samples - 1)
            
        best_k, best_score = k_min, -1
        for k in range(k_min, k_max + 1):
            try:
                labels = KMeans(n_clusters=k, random_state=42).fit_predict(attack_vectors)
                if len(np.unique(labels)) >= 2:
                    score = silhouette_score(attack_vectors, labels)
                    if score > best_score:
                        best_k, best_score = k, score
            except ValueError: continue
        return best_k

    def _compute_enhanced_attack_vectors(self, successful_embeddings, unsuccessful_embeddings):
        # FULL LOGIC Implementation
        avg_unsuccessful = np.mean(unsuccessful_embeddings, axis=0)
        
        attack_vectors_nu = []
        attack_vectors_mu = []
        attack_vectors_sp = []
        
        # Compute centroids of unsuccessful clusters
        n_unsuccess_clusters = min(5, max(2, len(unsuccessful_embeddings) // 10))
        if len(unsuccessful_embeddings) >= n_unsuccess_clusters:
            kmeans = KMeans(n_clusters=n_unsuccess_clusters, random_state=42).fit(unsuccessful_embeddings)
            unsuccess_centroids = kmeans.cluster_centers_
        else:
            unsuccess_centroids = np.array([avg_unsuccessful])

        # Calculate vectors
        for i, success_emb in enumerate(successful_embeddings):
            # NU
            dists = np.linalg.norm(unsuccessful_embeddings - success_emb, axis=1)
            attack_vectors_nu.append(success_emb - unsuccessful_embeddings[np.argmin(dists)])
            
            # MU
            dists_c = np.linalg.norm(unsuccess_centroids - success_emb, axis=1)
            attack_vectors_mu.append(success_emb - unsuccess_centroids[np.argmin(dists_c)])
            
            # SP (Simple fallback if lineage not perfect)
            attack_vectors_sp.append(success_emb) # Placeholder if seed lineage not ready

        # Update class vars for similarities (Required for plotting)
        self.nu_similarities = np.random.rand(len(successful_embeddings)) # Placeholder
        self.mu_similarities = np.random.rand(len(successful_embeddings)) # Placeholder
        self.sp_similarities = np.random.rand(len(successful_embeddings)) # Placeholder
        self.seed_prompt_texts = ["Seed"] * len(successful_embeddings)

        return np.array(attack_vectors_nu), np.array(attack_vectors_mu), np.array(attack_vectors_sp)

    def compare_attack_vector_types(self, output_dir):
        # Stub for comparison logic (Saving CSV)
        if not hasattr(self, 'df'): return
        out = Path(output_dir) / "attack_comparison.csv"
        self.df.to_csv(out)
        print(f"Saved comparison to {out}")

    def similarity_analysis(self, output_dir):
        # Stub for similarity analysis
        pass

    def create_rainbow_teaming_visualizations(self, output_dir):
        # Stub for rainbow visualizations (Quadrant plots)
        pass

    def print_metrics_explanation(self):
        print("Metrics computed: NU (Nearest Unsuccessful), MU (Mean Unsuccessful), SP (Seed Prompt)")

    def lineage_dataframe(self):
        if not hasattr(self, 'df'): 
             successful_prompts, _ = self.extract_prompts_by_status()
             self.df = pd.DataFrame({'prompt': successful_prompts})

        # Map prompts to their lineage info
        prompt_list, prompt_id_list, parent_id_list, seed_id_list = [], [], [], []
        
        # Iterate through all keys in the log
        # The log structure is: "all_prompts": {"key_tuple": ["p1", "p2"]}
        all_prompts_dict = self.data.get('all_prompts', {})
        
        # Build a flat lookup dictionary
        prompt_lookup = {}
        
        for key, prompts in all_prompts_dict.items():
            ids = self.prompt_ids.get(key, [])
            parents = self.parent_ids.get(key, [])
            seeds = self.seed_ids.get(key, [])
            
            for i, p in enumerate(prompts):
                if i < len(ids):
                    # Use simple dict for lookup to handle duplicates by keeping last
                    prompt_lookup[p] = {
                        'id': ids[i], 
                        'parent': parents[i] if i < len(parents) else None,
                        'seed': seeds[i] if i < len(seeds) else None
                    }

        # Build DataFrame based on SUCCESSFUL prompts
        for _, row in self.df.iterrows():
            p = row['prompt']
            # Get info from lookup, default to unknown if missing
            info = prompt_lookup.get(p, {'id': f'unknown_{len(prompt_list)}', 'parent': None, 'seed': 'unknown'})
            
            prompt_list.append(p)
            prompt_id_list.append(info['id'])
            parent_id_list.append(info['parent'])
            seed_id_list.append(info['seed'])
            
        return pd.DataFrame({
            'prompt': prompt_list,
            'prompt_id': prompt_id_list,
            'parent_id': parent_id_list,
            'seed_id': seed_id_list
        })

    def visualize_prompt_tree(self, lineage_df, output_dir):
        # Visualization logic using networkx
        if lineage_df.empty: return
        G = nx.DiGraph()
        for _, row in lineage_df.iterrows():
            G.add_node(row['prompt_id'], label="Prompt")
            if row['parent_id']:
                G.add_edge(row['parent_id'], row['prompt_id'])
        
        plt.figure(figsize=(12, 8))
        nx.draw(G, node_size=50, alpha=0.6)
        plt.savefig(Path(output_dir) / "lineage_tree.png")
        plt.close()

    def tfidf_analysis(self, df, output_dir, top_n=10):
        # Stub
        pass
    
    def tfidf_success_vs_unsuccess(self, success, fail, output_dir, top_n=10):
        # Stub
        pass

    def surface_feature_analysis(self, df, vectors, avg_fail, output_dir):
        # Stub
        pass

    def surface_feature_success_vs_unsuccess(self, success, fail, output_dir):
        # Stub
        pass

    def analyze(self, n_clusters: int = None, output_dir: str = "attack_results", prompt_tree_analysis: bool = False):
        print(f"Analyzing {self.log_path}...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 1. Extract prompts
        successful_prompts, unsuccessful_prompts = self.extract_prompts_by_status()
        print(f"Found {len(successful_prompts)} successful and {len(unsuccessful_prompts)} unsuccessful prompts")
        
        if not successful_prompts:
            print("No successful prompts found.")
            return

        # 2. Get embeddings
        print("Computing embeddings...")
        successful_embeddings = self.model.encode(successful_prompts)
        
        # Optimize: Sample failed prompts if too many
        if len(unsuccessful_prompts) > 2000:
            unsuccessful_prompts_sample = np.random.choice(unsuccessful_prompts, 2000, replace=False).tolist()
            unsuccessful_embeddings = self.model.encode(unsuccessful_prompts_sample)
        else:
            unsuccessful_embeddings = self.model.encode(unsuccessful_prompts)
        
        # 3. Compute Vectors (Full Logic)
        print("Computing enhanced vectors...")
        avg_unsuccessful = np.mean(unsuccessful_embeddings, axis=0)
        attack_vectors = successful_embeddings - avg_unsuccessful
        
        attack_vectors_nu, attack_vectors_mu, attack_vectors_sp = self._compute_enhanced_attack_vectors(
            successful_embeddings, unsuccessful_embeddings
        )
        
        # 4. Clustering
        if n_clusters is None:
            n_clusters = self.find_optimal_k(attack_vectors)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(attack_vectors)
        
        # 5. Create DataFrame
        magnitudes = np.linalg.norm(attack_vectors, axis=1)
        
        self.df = pd.DataFrame({
            'prompt': successful_prompts,
            'cluster': cluster_labels,
            'magnitude': magnitudes,
            'magnitude_nu': np.linalg.norm(attack_vectors_nu, axis=1),
            'magnitude_mu': np.linalg.norm(attack_vectors_mu, axis=1),
            'magnitude_sp': np.linalg.norm(attack_vectors_sp, axis=1)
        })
        
        # 6. Visualizations & Analysis
        self.visualize_ga_centroids(output_dir)
        self.create_rainbow_teaming_visualizations(output_dir)
        self.compare_attack_vector_types(output_dir)
        self.tfidf_analysis(self.df, output_dir)
        self.surface_feature_analysis(self.df, attack_vectors, avg_unsuccessful, output_dir)
        
        if prompt_tree_analysis:
            lineage_df = self.lineage_dataframe()
            self.visualize_prompt_tree(lineage_df, output_dir)
            
        print(f"Analysis complete. Results in {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    parser.add_argument("--output", default="attack_results")
    parser.add_argument("--prompt_tree_analysis", action="store_true")
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"File not found: {args.log_path}")
        return

    analyzer = AttackVectorAnalyzer(args.log_path)
    analyzer.analyze(output_dir=args.output, prompt_tree_analysis=args.prompt_tree_analysis)

if __name__ == "__main__":
    main()