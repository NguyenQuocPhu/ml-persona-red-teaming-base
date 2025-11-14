import heapq
from typing import List, Dict, Any
class AttackMemory:
    def __init__(self, k_top: int, k_bot: int):
        self.k_top = k_top
        self.k_bot = k_bot
        self.entries_top: list[dict] = []
        self.entries_bot: list[dict] = []
    
    def add(self, score: int, prompt: str, type: int):
        new_entry = {
            "score": score,
            "prompt": prompt
        }
        if type:
            self.entries_top.append(new_entry)
        else:
            self.entries_bot.append(new_entry)        
        key_function = lambda entry: entry["score"]
        k_top_items = heapq.nlargest(self.k_top, self.entries_top, key = key_function)
        k_bot_items = heapq.nsmallest(self.k_bot, self.entries_bot, key = key_function)

        self.entries_top = k_top_items
        self.entries_bot = k_bot_items
    
    def add_list(self, new_entries_list: List[Dict[str, Any]], entry_type: int):
        for entry in new_entries_list:
            # Gọi hàm add gốc, truyền score, prompt và type
            self.add(
                score=entry["score"], 
                prompt=entry["prompt"], 
                type=entry_type
            )

    def get_prompts_top(self, _size: int):
        key_function = lambda entry: entry["score"]
        size = min(self.k_top, len(self.entries_top))
        size = min(size, _size)
        return heapq.nlargest(size, self.entries_top, key = key_function)
    def get_prompts_bot(self, _size: int):
        key_function = lambda entry: entry["score"]
        size = min(self.k_bot, len(self.entries_bot))
        size = min(size, _size)
        return heapq.nsmallest(size, self.entries_bot, key = key_function)
    