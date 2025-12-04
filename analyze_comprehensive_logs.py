import random
import numpy as np
import argparse
import json
import glob
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
# --- Helper Functions ---

def load_comprehensive_log(log_path):
    """Load a comprehensive log file."""
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_self_bleu(prompts_list, n=4):
    """Tính điểm Self-BLEU. Lower score = More diverse."""
    if len(prompts_list) < 2:
        return 0.0

    try:
        tokenized_prompts = [word_tokenize(p.lower()) for p in prompts_list]
    except (LookupError, NameError):
        tokenized_prompts = [p.lower().split() for p in prompts_list]

    smoother = SmoothingFunction().method1
    scores = []
    weights = tuple([1.0/n] * n)

    for i, candidate in enumerate(tokenized_prompts):
        references = tokenized_prompts[:i] + tokenized_prompts[i+1:]
        if not references: continue
        
        score = sentence_bleu(references, candidate, weights=weights, smoothing_function=smoother)
        scores.append(score)

    return np.mean(scores) if scores else 0.0

def calculate_top_k_fitness_stats(evaluations, k=100, strategy='highest'):
    """Tính điểm Fitness trung bình và độ lệch chuẩn."""
    if not evaluations:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    # Đảm bảo k là số nguyên và không vượt quá số lượng dữ liệu
    k = min(int(k), len(evaluations))
    
    if strategy == 'highest':
        selected_evals = sorted(evaluations, key=lambda x: x.get('fitness', 0), reverse=True)[:k]
    elif strategy == 'random':
        selected_evals = random.sample(evaluations, k)
    else:
        raise ValueError("Strategy must be 'highest' or 'random'")
        
    fitness_scores = [ev.get('fitness', 0) for ev in selected_evals]
    
    if not fitness_scores:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": float(np.mean(fitness_scores)),
        "std": float(np.std(fitness_scores)),
        "min": float(np.min(fitness_scores)),
        "max": float(np.max(fitness_scores))
    }

def analyze_diversity_stability(all_prompts, method="Name", sample_size=100, num_iterations=10, n_gram=4):
    """Bootstrap sampling để tính Diversity stability."""
    actual_sample_size = min(sample_size, len(all_prompts))
    if actual_sample_size == 0:
        return {"mean": 0.0, "std": 0.0}

    bleu_scores = []
    print(f"--- Analyzing Diversity for {method} ---")
    
    for i in range(num_iterations):
        sample = random.sample(all_prompts, actual_sample_size)
        score = calculate_self_bleu(sample, n=n_gram)
        bleu_scores.append(score)
        
    mean_score = np.mean(bleu_scores)
    std_dev = np.std(bleu_scores)
    
    print(f"Result -> Mean Self-BLEU: {mean_score:.4f} | Std Dev: {std_dev:.4f}")
    return {"name": method, "mean": mean_score, "std": std_dev}

def compare_methods_diversity(persona_prompts, archive_prompts, sample_size=100, num_iterations=20):
    stats_persona = analyze_diversity_stability(persona_prompts, "PersonaTeaming", sample_size, num_iterations)
    stats_archive = analyze_diversity_stability(archive_prompts, "GrowingArchive", sample_size, num_iterations)
    
    print("\n====== COMPARISON CONCLUSION ======")
    print(f"PersonaTeaming Mean Self-BLEU: {stats_persona['mean']:.4f} (±{stats_persona['std']:.4f})")
    print(f"GrowingArchive Mean Self-BLEU: {stats_archive['mean']:.4f} (±{stats_archive['std']:.4f})")
    
    diff = stats_archive['mean'] - stats_persona['mean']
    if diff > 0:
        print(f"-> PersonaTeaming is MORE DIVERSE by {diff:.4f} points (Lower Self-BLEU is better).")
    elif diff < 0:
        print(f"-> GrowingArchive is MORE DIVERSE by {abs(diff):.4f} points.")
    else:
        print("-> Both methods have equal diversity.")

def plot_fitness_distribution(scores_ga, scores_pt, output_filename="fitness_dist.png"):
    """
    Vẽ biểu đồ phân bố điểm số (Fitness) của 2 phương pháp.
    
    Args:
        scores_ga (list): Danh sách điểm số của Growing Archive (VD: [0.9, 0.8, 1.0...])
        scores_pt (list): Danh sách điểm số của Persona Teaming
    """
    print("\n[Viz] Đang vẽ biểu đồ phân bố Fitness...")

    # BƯỚC 1: Chuẩn bị dữ liệu cho "ngăn nắp"
    # Chúng ta tạo một cái bảng (DataFrame) để máy dễ vẽ
    # Cột 1: Phương pháp (Tên lớp học)
    # Cột 2: Điểm số (Fitness)
    data = []
    
    for s in scores_ga:
        data.append({"Method": "GrowingArchive", "Fitness": s})
    
    for s in scores_pt:
        data.append({"Method": "PersonaTeaming", "Fitness": s})
        
    df = pd.DataFrame(data)

    # BƯỚC 2: Vẽ biểu đồ
    plt.figure(figsize=(10, 6)) # Tạo khung tranh kích thước 10x6
    
    # Dùng hàm histplot (Histogram Plot)
    # x="Fitness": Trục ngang là điểm số
    # hue="Method": Tô màu khác nhau cho từng phương pháp
    # kde=True: Vẽ thêm đường cong mềm mại để dễ nhìn xu hướng
    # bins=20: Chia điểm số thành 20 cái giỏ nhỏ
    sns.histplot(
        data=df, 
        x="Fitness", 
        hue="Method", 
        kde=True, 
        bins=20,
        palette={'PersonaTeaming': '#FF4B4B', 'GrowingArchive': '#1F9D89'}, # Màu Đỏ vs Xanh
        alpha=0.6 # Độ trong suốt (để nếu chồng lên nhau vẫn nhìn thấy)
    )
    
    # Trang trí cho đẹp
    plt.title("So sánh Phân Bố Điểm Số (Fitness Distribution)", fontsize=15)
    plt.xlabel("Điểm Fitness (Càng cao càng tốt)", fontsize=12)
    plt.ylabel("Số lượng Prompt (Tần suất)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3) # Kẻ dòng mờ mờ cho dễ dóng
    
    # BƯỚC 3: Lưu thành file ảnh (để xem trên Kaggle)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Đã lưu biểu đồ vào file: {output_filename}")
    plt.close() # Đóng lại cho đỡ tốn RAM


def find_log_files(directory):
    directory = Path(directory)
    
    # --- 1. Find GA Log ---
    comprehensive_ga_log = None
    ga_global = directory / "comprehensive_ga_log_global.json"
    
    if ga_global.exists():
        comprehensive_ga_log = str(ga_global)
    else:
        # Check patterns if global file doesn't exist
        ga_pattern = directory / "comprehensive_ga_log_*.json"
        ga_candidates = sorted(list(glob.glob(str(ga_pattern))))
        if ga_candidates:
            comprehensive_ga_log = ga_candidates[-1]

    # --- 2. Find Standard Log ---
    comprehensive_log = None
    std_global = directory / "comprehensive_log_global.json"
    
    if std_global.exists():
        comprehensive_log = str(std_global)
    else:
        # Check patterns
        std_pattern = directory / "comprehensive_log_*.json"
        std_candidates = sorted(list(glob.glob(str(std_pattern))))
        if std_candidates:
            comprehensive_log = std_candidates[-1]

    return comprehensive_ga_log, comprehensive_log

def main():
    parser = argparse.ArgumentParser(description="Analyze comprehensive RainbowPlus logs")
    # SỬA: Thêm type=int
    parser.add_argument("--top-k", type=int, default=100, help="Number of top-k to compare 2 methods")
    parser.add_argument("directory", help="Directory containing log files")
    parser.add_argument("--output", help="Output file for analysis results")
    args = parser.parse_args()

    try:
        comprehensive_ga_log, comprehensive_log = find_log_files(args.directory)
        print(f"Standard Log: {comprehensive_log}")
        print(f"GA Log:       {comprehensive_ga_log}")
        print("-" * 40)
            
    except Exception as e:
        print(f"Error finding logs: {e}")
        return
    
    ga_prompts = []
    ga_scores = []
    if comprehensive_ga_log:
        try:
            # SỬA: Load trực tiếp file, không load chồng chéo
            ga_data = load_comprehensive_log(comprehensive_ga_log)
            
            ga_state = ga_data.get('ga_state', {})
            elites = ga_state.get('elites', {})
            
            ga_evaluations = []
            for key, list_of_elite in elites.items():
                if list_of_elite:
                    max_evaluation = max(list_of_elite, key=lambda x: x.get('fitness', 0))
                    ga_evaluations.append(max_evaluation)
                    # Take the highest fitness in each cells
                    for elite in list_of_elite:
                        ga_prompts.append(elite['prompt'])
                        ga_scores.append(elite['fitness'])
            
            print(f"[GrowingArchive] Loaded {len(ga_prompts)} prompts.")
            
            stats = calculate_top_k_fitness_stats(ga_evaluations, args.top_k, 'highest')
            print(f"GA Top-{args.top_k} Fitness: {stats['mean']:.4f} (±{stats['std']:.4f})")
            
        except Exception as e:
            print(f"Error processing GA log: {e}")

    # --- PROCESS STANDARD LOGS ---
    persona_prompts = []
    persona_scores = []
    if comprehensive_log:
        try:
            std_data = load_comprehensive_log(comprehensive_log)
            all_prompts = std_data.get('all_prompts', {})
            all_scores = std_data.get('all_scores', {})
            
            persona_evaluations = []
            for category, prompts in all_prompts.items():
                persona_prompts.extend(prompts)
                scores = all_scores.get(category, [])
                
                for prompt, score in zip(prompts, scores):
                    persona_evaluations.append({
                        "prompt": prompt,
                        "fitness": score,
                    })
                    persona_scores.append(score)
            
            print(f"\n[PersonaTeaming] Loaded {len(persona_prompts)} prompts.")
            
            # Highest Stats
            stats_high = calculate_top_k_fitness_stats(persona_evaluations, args.top_k, 'highest')
            print(f"Persona Highest Top-{args.top_k} Fitness: {stats_high['mean']:.4f} (±{stats_high['std']:.4f})")
            
            # Random Stats
            stats_rand = calculate_top_k_fitness_stats(persona_evaluations, args.top_k, 'random')
            print(f"Persona Random Top-{args.top_k} Fitness:  {stats_rand['mean']:.4f} (±{stats_rand['std']:.4f})")
            
        except Exception as e:
            print(f"Error processing Standard log: {e}")
    
    print("-" * 40)

    compare_methods_diversity(persona_prompts, ga_prompts)

    # 3. Compute Embeddings
    print("\n[AI] Computing Embeddings (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ga_emb = model.encode(ga_prompts)
    pt_emb = model.encode(persona_prompts)
    plot_fitness_distribution(ga_scores, persona_scores)

if __name__ == "__main__":
    main()