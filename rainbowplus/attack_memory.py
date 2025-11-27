import heapq
from typing import List, Dict, Any
import random
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
            "score": score,
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

    def get_prompts_top(self, sample_size: int) -> List[Dict[str, Any]]:
        """
        Lấy ngẫu nhiên sample_size phần tử từ danh sách Top.
        """
        if not self.entries_top:
            return []
        
        # Đảm bảo không lấy quá số lượng đang có
        k = min(sample_size, len(self.entries_top))
        return random.sample(self.entries_top, k)

    def get_prompts_bot(self, sample_size: int) -> List[Dict[str, Any]]:
        """
        Lấy ngẫu nhiên sample_size phần tử từ danh sách Bot.
        """
        if not self.entries_bot:
            return []
            
        # Đảm bảo không lấy quá số lượng đang có
        k = min(sample_size, len(self.entries_bot))
        return random.sample(self.entries_bot, k)

    
    