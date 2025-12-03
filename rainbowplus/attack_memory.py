import heapq
from typing import List, Dict, Any
import random
import numpy as np

class AttackMemory:
    def __init__(self, threshold_bot: float, threshold_top: float):
        self.threshold_bot = threshold_bot
        self.threshold_top = threshold_top
        self.entries_top: list[dict] = []
        self.entries_bot: list[dict] = []
    
    def add(self, entry: Dict[str, Any]):
        """
        Thêm một entry mới.
        """
        score = entry.get("score", 0)
        
        clean_entry = {
            "behavior": entry.get("behavior", None),
            # Thêm key fitness để đồng bộ với hàm sort bên dưới
            "fitness": score,
            "score": score,
            "prompt": entry.get("prompt", ""),
            "persona": entry.get("persona", {}),
            "risk": entry.get("risk", ""),
            "style": entry.get("style", "")
        }

        # --- LOGIC XỬ LÝ TOP (SUCCESSFUL) ---
        if score >= self.threshold_top:
            self.entries_top.append(clean_entry)

        # --- LOGIC XỬ LÝ BOT (FAILED) ---
        elif score <= self.threshold_bot:
            self.entries_bot.append(clean_entry)
    
    def add_list(self, new_entries_list: List[Dict[str, Any]]):
        for entry in new_entries_list:
            self.add(entry)

    def _calculate_cosine_distance_matrix(self, A, B):
        """
        Tính ma trận khoảng cách Cosine giữa tập A và tập B.
        Distance = 1 - CosineSimilarity
        """
        # Chuẩn hóa vector để tính cosine (chia cho độ dài vector)
        # Cộng 1e-9 để tránh chia cho 0
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        
        # Tính similarity: A . B.T
        similarity = np.dot(A_norm, B_norm.T)
        
        # Distance = 1 - Similarity (Clip để tránh sai số dấu phẩy động)
        return 1.0 - np.clip(similarity, -1.0, 1.0)

    def _select_diverse_subset(self, candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        Thuật toán Greedy Max-Sum Distance.
        """
        if not candidates:
            return []
        
        # 1. Lọc các candidate hợp lệ (có behavior vector)
        valid_candidates = [
            c for c in candidates 
            if c.get('behavior') is not None and len(c.get('behavior', [])) > 0
        ]
        
        # Nếu không đủ dữ liệu có vector, trả về random hoặc lấy hết
        if len(valid_candidates) < k:
            return valid_candidates

        # 2. Sắp xếp theo Fitness giảm dần
        valid_candidates.sort(key=lambda x: x.get('fitness', 0.0), reverse=True)
        
        try:
            embeddings = np.array([item['behavior'] for item in valid_candidates])
        except Exception as e:
            # Fallback nếu lỗi dữ liệu
            return random.sample(candidates, k)

        first_idx = random.randint(0, len(valid_candidates) - 1)
        selected_indices = [first_idx]

        # Tạo danh sách index còn lại dựa trên độ dài của valid_candidates
        candidate_indices = [i for i in range(len(valid_candidates)) if i != first_idx]

        # 4. Vòng lặp Greedy
        while len(selected_indices) < k and candidate_indices:
            # Lấy vector
            selected_vecs = embeddings[selected_indices]
            candidate_vecs = embeddings[candidate_indices]

            # Tính ma trận khoảng cách: (num_candidates x num_selected)
            dist_matrix = self._calculate_cosine_distance_matrix(candidate_vecs, selected_vecs)

            # Tính tổng khoảng cách tới tất cả các điểm đã chọn (Max-Sum)
            mean_dists = np.sum(dist_matrix, axis=1)

            # Chọn candidate có tổng khoảng cách lớn nhất
            best_candidate_idx_in_pool = np.argmax(mean_dists)
            
            # Map ngược lại index gốc trong danh sách valid_candidates
            original_idx = candidate_indices[best_candidate_idx_in_pool]
            
            # Cập nhật danh sách
            selected_indices.append(original_idx)
            
            # Xóa khỏi danh sách ứng viên (dùng pop hoặc del theo index trong pool)
            # Lưu ý: candidate_indices và mean_dists đồng bộ index với nhau
            del candidate_indices[best_candidate_idx_in_pool]

        # Trả về kết quả
        return [valid_candidates[i] for i in selected_indices]

    def get_prompts_top(self, sample_size: int) -> List[Dict[str, Any]]:
        return self._select_diverse_subset(self.entries_top, sample_size)

    def get_prompts_bot(self, sample_size: int) -> List[Dict[str, Any]]:
        return self._select_diverse_subset(self.entries_bot, sample_size)