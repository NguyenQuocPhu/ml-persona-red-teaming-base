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

def calculate_distinct_n(prompts_list, n=2):
    """
    (Bổ sung) Tính Distinct-N: Tỷ lệ N-gram độc nhất.
    Metric này bổ trợ rất tốt cho Self-BLEU.
    Cao = Tốt (Từ vựng phong phú).
    """
    if not prompts_list: return 0.0
    ngrams_all = []
    for p in prompts_list:
        words = p.lower().split()
        if len(words) < n: continue
        ngrams_all.extend([tuple(words[i:i+n]) for i in range(len(words)-n+1)])
    
    if not ngrams_all: return 0.0
    return len(set(ngrams_all)) / len(ngrams_all)



def get_top_k_data(evaluations, k=100, strategy='highest'):
    """Lấy Top-K theo fitness."""
    if not evaluations: return []
    k = min(int(k), len(evaluations))
    
    if strategy == 'highest':
        # Sort giảm dần và lấy K cái đầu
        selected = sorted(evaluations, key=lambda x: x.get('fitness', 0), reverse=True)[:k]
    elif strategy == 'random':
        # (Ít dùng cho Self-BLEU Top-K nhưng vẫn giữ)
        selected = random.sample(evaluations, k)
    else:
        return []
        
    return selected


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

def analyze_convex_hull_simple(embeddings, method_name="Method"):
    """
    1. Tính thể tích ước lượng (dựa trên PCA 3D).
    2. Vẽ hình bao lồi 2D (dựa trên PCA 2D).
    """
    # --- 1. TÍNH TOÁN (Metric) ---
    # Giảm xuống 3 chiều để tính thể tích "đại diện"
    if len(embeddings) >= 4:
        pca_3d = PCA(n_components=3)
        emb_3d = pca_3d.fit_transform(embeddings)
        volume = ConvexHull(emb_3d).volume
    else:
        volume = 0.0
    
    print(f"[{method_name}] Pseudo-Volume (3D PCA): {volume:.4f}")

    # --- 2. VẼ HÌNH 2D (Visualization) ---
    # Giảm xuống 2 chiều để vẽ lên giấy
    if len(embeddings) >= 3:
        pca_2d = PCA(n_components=2)
        points_2d = pca_2d.fit_transform(embeddings)
        hull = ConvexHull(points_2d)

        # Vẽ các điểm
        plt.figure(figsize=(8, 6))
        plt.plot(points_2d[:,0], points_2d[:,1], 'o', color='blue', alpha=0.3, label='Prompts')

        # Vẽ đường bao lồi (Sợi dây chun)
        for simplex in hull.simplices:
            plt.plot(points_2d[simplex, 0], points_2d[simplex, 1], 'r-', lw=2) # Đường màu đỏ
        
        # Tô màu vùng bên trong (Option)
        plt.fill(points_2d[hull.vertices,0], points_2d[hull.vertices,1], 'r', alpha=0.1)

        plt.title(f'{method_name} - 2D Convex Hull Coverage')
        plt.xlabel('PCA Dim 1')
        plt.ylabel('PCA Dim 2')
        plt.grid(True, linestyle=':', alpha=0.5)
        
        # Lưu file
        filename = f"hull_2d_{method_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu ảnh 2D đơn giản: {filename}")
        plt.close()
    else:
        print("Không đủ điểm dữ liệu để vẽ Hull.")

    return volume

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
    
    centroids = []
    # 3. Compute Embeddings
    print("\n[AI] Computing Embeddings (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    ga_prompts = []
    ga_scores = []
    if comprehensive_ga_log:
        try:
            # SỬA: Load trực tiếp file, không load chồng chéo
            ga_data = load_comprehensive_log(comprehensive_ga_log)
            
            ga_state = ga_data.get('ga_state', {})
            centroids = ga_state.get('centroids', [])
            print(ga_state.get('n_centroids', 0))
            print(f"#######################")
            elites = ga_state.get('elites', {})
            
            ga_evaluations = []
            for key, list_of_elite in elites.items():
                if list_of_elite:
                    max_evaluation = max(list_of_elite, key=lambda x: x.get('fitness', 0))
                    #ga_evaluations.append(max_evaluation)
                    # Take the highest fitness in each cells
                    for elite in list_of_elite:
                        ga_prompts.append(elite['prompt'])
                        ga_scores.append(elite['fitness'])
                        ga_evaluations.append(elite)
            
            print(f"[GrowingArchive] Loaded {len(ga_prompts)} prompts.")
            
            stats = get_top_k_data(ga_evaluations, args.top_k, 'highest')
            scores = [e['fitness'] for e in stats]
            print(f"GA Top-{args.top_k} Fitness: {float(np.mean(scores)):.4f} (±{float(np.std(scores)):.4f}")
            flat_prompts = [e['prompt'] for e in stats]
            diversity = calculate_self_bleu(flat_prompts)
            print(f"GA Self-BLEU: {diversity:.4f}")
            distinct_n = calculate_distinct_n(flat_prompts)
            print(f"GA Distinct-2: {distinct_n:.4f}")
            ga_emb = model.encode(flat_prompts)
            analyze_convex_hull_simple(ga_emb, "GrowingArchive")
            
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
            stats_high = get_top_k_data(persona_evaluations, args.top_k, 'highest')
            scores = [e['fitness'] for e in stats_high]
            print(f"Persona Highest Top-{args.top_k} Fitness: {float(np.mean(scores)):.4f} (±{float(np.std(scores)):.4f}")
            flat_prompts = [e['prompt'] for e in stats_high]
            diversity = calculate_self_bleu(flat_prompts)
            print(f"Persona Self-BLEU: {diversity:.4f}")
            distinct_n = calculate_distinct_n(flat_prompts)
            print(f"Persona Distinct-2: {distinct_n:.4f}")
            pt_emb = model.encode(flat_prompts)
            analyze_convex_hull_simple(pt_emb, "PersonaTeaming")
            '''
            # Random Stats
            stats_rand = get_top_k_data(persona_evaluations, args.top_k, 'random')
            print(f"Persona Random Top-{args.top_k} Fitness:  {stats_rand['mean']:.4f} (±{stats_rand['std']:.4f})")
            '''
        except Exception as e:
            print(f"Error processing Standard log: {e}")
    
    print("-" * 40)


    plot_fitness_distribution(ga_scores, persona_scores)
    analyze_convex_hull_simple(centroids, "Centroids")

if __name__ == "__main__":
    main()
    