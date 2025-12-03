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
        
        Args:
            entry: Dict chứa keys {'score', 'prompt', 'persona', 'risk', 'style'}
            threshold_bot: Nếu score <= giá trị này -> Thêm vào danh sách thất bại (Bot)
            threshold_top: Nếu score >= giá trị này -> Thêm vào danh sách thành công (Top)
        """
        score = entry.get("score", 0)
        
        # Chuẩn hóa dữ liệu để đảm bảo đủ trường thông tin cho việc tạo prompt sau này
        clean_entry = {
            "behavior": entry.get("behavior", None),
            "fitness": score,
            "prompt": entry.get("prompt", ""),
            "persona": entry.get("persona", {}), # Persona dạng dict hoặc yaml string
            "risk": entry.get("risk", ""),
            "style": entry.get("style", "")
        }

        # --- LOGIC XỬ LÝ TOP (SUCCESSFUL) ---
        # Chỉ lưu nếu score vượt qua ngưỡng top
        if score >= self.threshold_top:
            self.entries_top.append(clean_entry)

        # --- LOGIC XỬ LÝ BOT (FAILED) ---
        # Chỉ lưu nếu score thấp hơn hoặc bằng ngưỡng bot
        elif score <= self.threshold_bot:
            self.entries_bot.append(clean_entry)
    
    def add_list(self, new_entries_list: List[Dict[str, Any]]):
        """
        Thêm danh sách nhiều entry cùng lúc.
        """
        for entry in new_entries_list:
            self.add(entry)

    def _calculate_cosine_distance_matrix(self, A, B):
        """
        Tính ma trận khoảng cách Cosine giữa tập A và tập B.
        Distance = 1 - CosineSimilarity
        """
        # Chuẩn hóa vector để tính cosine (chia cho độ dài vector)
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        
        # Tính similarity: A . B.T
        similarity = np.dot(A_norm, B_norm.T)
        
        # Distance = 1 - Similarity
        return 1.0 - similarity

    def _select_diverse_subset(self, candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """
        Thuật toán Greedy Max-Sum Distance:
        1. Chọn phần tử có Fitness cao nhất đầu tiên.
        2. Các phần tử tiếp theo được chọn sao cho tổng khoảng cách (sự khác biệt) 
           của nó so với các phần tử ĐÃ CHỌN là lớn nhất.
        """
        if not candidates:
            return []
        
        # Nếu số lượng cần lấy >= số lượng có sẵn, trả về tất cả
        if k >= len(candidates):
            return candidates

        # 1. Chuẩn bị dữ liệu
        # Sắp xếp giảm dần theo fitness để đảm bảo phần tử đầu tiên luôn là "Elite" tốt nhất
        # (Giả sử key chứa điểm là 'fitness' hoặc 'score')
        candidates = sorted(candidates, key=lambda x: x.get('fitness', 0.0), reverse=True)
        
        # Trích xuất vector behavior (embedding) ra mảng numpy để tính toán cho nhanh
        # Lưu ý: Cần đảm bảo key 'behavior' tồn tại và là list/array
        try:
            embeddings = np.array([item['behavior'] for item in candidates])
        except KeyError:
            # Fallback: Nếu không có embedding, quay về random
            return random.sample(candidates, k)

        # 2. Khởi tạo
        selected_indices = random.randint(0, len(candidates))
        candidate_indices = [i for i in range(len(candidates)) if i != selected_indices] # Các phần tử còn lại

        # 3. Vòng lặp Greedy (Tham lam)
        while len(selected_indices) < k and candidate_indices:
            # Lấy vector của những cái đã chọn và những cái chưa chọn
            selected_vecs = embeddings[selected_indices]
            candidate_vecs = embeddings[candidate_indices]

            # Tính khoảng cách từ mỗi candidate đến TẤT CẢ các selected
            # Shape: (num_candidates, num_selected)
            dist_matrix = self._calculate_cosine_distance_matrix(candidate_vecs, selected_vecs)

            # Tính tổng khoảng cách cho mỗi candidate (Sum Distance)
            # Axis 1: cộng dồn khoảng cách tới từng phần tử đã chọn
            mean_dists = np.sum(dist_matrix, axis=1)

            # Chọn candidate có tổng khoảng cách lớn nhất (Khác biệt nhất với tập đã chọn)
            best_candidate_idx_in_pool = np.argmax(mean_dists)
            
            # Map ngược lại index gốc
            original_idx = candidate_indices[best_candidate_idx_in_pool]
            
            # Cập nhật danh sách
            selected_indices.append(original_idx)
            del candidate_indices[best_candidate_idx_in_pool]

        # Trả về các phần tử tương ứng với index đã chọn
        return [candidates[i] for i in selected_indices]

    def get_prompts_top(self, sample_size: int) -> List[Dict[str, Any]]:
        """
        Lấy sample_size phần tử từ danh sách Top, ưu tiên sự ĐA DẠNG (Max-Sum).
        """
        return self._select_diverse_subset(self.entries_top, sample_size)

    def get_prompts_bot(self, sample_size: int) -> List[Dict[str, Any]]:
        """
        Lấy sample_size phần tử từ danh sách Bot, ưu tiên sự ĐA DẠNG (Max-Sum).
        """
        return self._select_diverse_subset(self.entries_bot, sample_size)
    
    