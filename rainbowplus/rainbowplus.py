#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import sys
import argparse
import random
import json
import time
import logging
import uuid  # Required for ID generation
from pathlib import Path
import yaml
import numpy as np
from typing import Dict, List
from copy import deepcopy

# --- Imports ---
from rainbowplus.scores import BleuScoreNLTK, OpenAIGuard, LlamaGuard
from rainbowplus.utils import (
    load_txt,
    load_json,
    initialize_language_models,
    save_ga_iteration_log,      # New logging function
    save_ga_comprehensive_log,  # New logging function
    format_example,
    save_attack_memory_log,
)
from rainbowplus.archive import Archive
from rainbowplus.configs import ConfigurationLoader
from rainbowplus.prompts import MUTATOR_PROMPT, TARGET_PROMPT
from rainbowplus.configs.base import LLMConfig
from rainbowplus.mutators.persona import PersonaMutator
from rainbowplus.attack_memory import AttackMemory
# Set fixed random seed
RANDOM_SEED = 15
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- UTILS FOR GROWING ARCHIVE ---
def cosine_distance_batch(X, Y):
    """Tính khoảng cách Cosine: 1 - Similarity."""
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    # Tránh chia cho 0
    X_norm[X_norm == 0] = 1e-12
    Y_norm[Y_norm == 0] = 1e-12
    
    dot_product = X @ Y.T
    similarity = dot_product / (X_norm @ Y_norm.T)
    # Clip để tránh lỗi sai số dấu phẩy động
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1.0 - similarity

def _compute_distance_to_centroids(b_vector, centroids):
    """Tìm centroid gần nhất và khoảng cách tới nó."""
    if len(centroids) == 0:
        return np.inf, -1
    if b_vector.ndim == 1:
        b_vector = b_vector.reshape(1, -1)
    distances = cosine_distance_batch(b_vector, centroids)
    c_id = np.argmin(distances)
    return distances[0, c_id], c_id

# --- MAIN CLASS: Dynamic Expansion Archive ---
class GrowingArchive:
    def __init__(self, 
                 max_cells: int, 
                 n_behavior_dim: int, 
                 fitness_threshold: float, 
                 delta: float, 
                 k_elites: int = 10):
        """
        Args:
            max_cells: Sức chứa tối đa (Soft Cap).
            n_behavior_dim: Số chiều embedding (384).
            fitness_threshold: Lọc rác.
            delta: Ngưỡng khoảng cách để xác định "Vùng đất mới".
            k_elites: Số lượng elite trong mỗi vùng.
        """
        self.max_cells = max_cells
        self.fitness_threshold = fitness_threshold
        self.delta = delta
        self.k_elites = k_elites
        
        # Centroids matrix (Cấp phát trước bộ nhớ tối đa)
        self.centroids = np.zeros((max_cells, n_behavior_dim), dtype=np.float32)
        
        # Số lượng centroid thực tế đang hoạt động
        self.n_centroids = 0
        
        # Kho lưu trữ: Map cell_id -> List of Dicts
        self.elites : Dict[int, List[Dict]] = {}
        
        # Cache cho Pruning (dùng khi đầy kho)
        self.c_id_neighbors = np.zeros((max_cells, max_cells), dtype=int)
        self.d_neighbors = np.zeros((max_cells, max_cells), dtype=np.float32)

    def _compute_pruning_metrics(self):
        """
        Chỉ tính toán khi cần Pruning (tìm hàng xóm gần nhất để xóa).
        Tính khoảng cách giữa các centroid đang hoạt động.
        """
        if self.n_centroids < 2:
            return

        active_centroids = self.centroids[:self.n_centroids]
        distances = cosine_distance_batch(active_centroids, active_centroids)
        np.fill_diagonal(distances, np.inf)
        
        # Lưu cache để dùng cho việc chọn nạn nhân khi đầy bộ nhớ
        self.c_id_neighbors[:self.n_centroids, :self.n_centroids] = np.argsort(distances, axis=1)
        
        rows = np.arange(self.n_centroids)[:, None]
        sorted_indices = self.c_id_neighbors[:self.n_centroids, :self.n_centroids]
        self.d_neighbors[:self.n_centroids, :self.n_centroids] = distances[rows, sorted_indices]

    def _add_to_cell_top_k(self, cell_id: int, evaluation: Dict):
        """Logic Top-K Multi-Elite."""
        if cell_id not in self.elites:
            self.elites[cell_id] = []
            
        self.elites[cell_id].append(deepcopy(evaluation))
        # Sort giảm dần theo fitness
        self.elites[cell_id].sort(key=lambda x: x.get("fitness", 0.0), reverse=True)
        # Giữ Top-K
        if len(self.elites[cell_id]) > self.k_elites:
            self.elites[cell_id] = self.elites[cell_id][:self.k_elites]

    def _migrate_and_compete(self, orphaned_elites: List[Dict]):
        """Cơ chế Tái định cư: Đưa dân cũ đi tìm nhà mới."""
        active_centroids = self.centroids[:self.n_centroids]
        for elite in orphaned_elites:
            b_vec = elite["behavior"].reshape(1, -1)
            dist, new_cell_id = _compute_distance_to_centroids(b_vec, active_centroids)
            self._add_to_cell_top_k(new_cell_id, elite)

    def add_evaluation(self, evaluation: Dict) -> str:
        """
        Logic chính: Expansion -> Replacement -> Competition.
        """
        if evaluation["fitness"] < self.fitness_threshold:
            return "rejected_fitness_low"

        b_vector = evaluation["behavior"].reshape(1, -1)
        active_centroids = self.centroids[:self.n_centroids]

        # --- TRƯỜNG HỢP 0: KHO RỖNG ---
        if self.n_centroids == 0:
            self.centroids[0] = b_vector
            self._add_to_cell_top_k(0, evaluation)
            self.n_centroids += 1
            return "added_init"

        # Tính khoảng cách tới centroid gần nhất
        d_to_nearest, closest_cell_id = _compute_distance_to_centroids(b_vector, active_centroids)

        # --- NHÁNH 1: EXPANSION CHECK (Kiểm tra Mở rộng) ---
        # Điều kiện: Đủ lạ (xa hơn delta)
        is_novel = d_to_nearest > self.delta

        if is_novel:
            # 1.1 Nếu còn chỗ trống (Chưa chạm Soft Cap) -> TẠO MỚI NGAY
            if self.n_centroids < self.max_cells:
                new_cell_id = self.n_centroids
                self.centroids[new_cell_id] = b_vector
                self._add_to_cell_top_k(new_cell_id, evaluation)
                self.n_centroids += 1
                
                # Cập nhật metric nếu sắp đầy (để chuẩn bị cho pruning)
                if self.n_centroids >= self.max_cells * 0.9: 
                    self._compute_pruning_metrics()
                    
                return "expanded_new_niche"
            
            # 1.2 Nếu đã đầy kho (Chạm Soft Cap) -> PHẢI THAY THẾ (Replacement)
            else:
                # Cần cập nhật metric để tìm kẻ yếu nhất
                self._compute_pruning_metrics()
                
                # Tìm khu vực chật chội nhất (2 centroid gần nhau nhất)
                c_A = np.argmin(self.d_neighbors[:, 0])
                c_B = self.c_id_neighbors[c_A, 0]
                
                # So sánh ai chật hơn (dựa vào hàng xóm thứ 2)
                if self.d_neighbors[c_A, 1] < self.d_neighbors[c_B, 1]:
                    pruned_id = c_A
                else:
                    pruned_id = c_B
                
                # Cứu hộ dân cũ
                orphans = self.elites.get(pruned_id, [])
                
                # Ghi đè Centroid mới vào vị trí cũ
                self.centroids[pruned_id] = b_vector
                self.elites[pruned_id] = [] # Reset nhà
                self._add_to_cell_top_k(pruned_id, evaluation)
                
                # Tái định cư dân cũ
                if orphans:
                    self._migrate_and_compete(orphans)
                
                return "replaced_novelty"

        # --- NHÁNH 2: COMPETITION CHECK (Cạnh tranh nội bộ) ---
        # Nếu không đủ lạ (d_to_nearest <= delta), nó thuộc về ô cũ
        else:
            # Thêm vào ô gần nhất và cạnh tranh Top-K
            current_list = self.elites[closest_cell_id]
            
            # Chỉ return success nếu nó thực sự lọt vào Top-K
            # (Logic này đã nằm trong hàm _add nhưng ta check để log cho đúng)
            if len(current_list) < self.k_elites or evaluation["fitness"] > current_list[-1]["fitness"]:
                self._add_to_cell_top_k(closest_cell_id, evaluation)
                return "competed_success"
            
            return "rejected_duplicate"

    def sample_parent(self):
        """Chọn ngẫu nhiên 1 ngách, sau đó chọn 1 elite."""
        if not self.elites:
            return None
        random_cell_id = random.choice(list(self.elites.keys()))
        elite_list = self.elites[random_cell_id]
        if not elite_list: return None
        return random.choice(elite_list)

# --- MAIN LOGIC ---

def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Prompt Generation")
    parser.add_argument("--num_samples", type=int, help="Number of initial seed prompts")
    parser.add_argument("--n_cells", type=int, default = 100, help="Number of cells in the archive")
    parser.add_argument("--max_iters", type=int, help="Maximum number of iteration steps")
    parser.add_argument("--sim_threshold", type=float, help="Similarity threshold for prompt mutation")
    parser.add_argument("--num_mutations", type=int, help="Number of prompt mutations per iteration")
    parser.add_argument("--fitness_threshold", type=float, help="Minimum fitness score to add prompt to archive")
    parser.add_argument("--config_file", type=str, default="./configs/base.yml", help="Path to configuration file")
    parser.add_argument("--log_dir", type=str, help="Directory for storing logs")
    parser.add_argument("--log_interval", type=int, help="Number of iterations between log saves")
    parser.add_argument("--seed_memory", type = str, help = "Path of seed prompts for memory")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--target_llm", type=str, help="Path to repository of target LLM")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle seed prompts")
    parser.add_argument("--threshold_bot_memory", type = float, default = 0.4, help = "Maximize fitness score to add prompt to memory")
    parser.add_argument("--threshold_top_memory", type = float, default = 0.6, help = "Minimize fitness score to add prompt to memory")
    parser.add_argument("--number_example_prompts", type = int, help = "Number of example prompts in few-shot")
    return parser.parse_args()

def merge_config_with_args(config, args):
    merged_args = type('Args', (), {})()
    merged_args.num_samples = config.num_samples
    merged_args.n_cells = config.n_cells
    merged_args.max_iters = config.max_iters
    merged_args.sim_threshold = config.sim_threshold
    merged_args.num_mutations = config.num_mutations
    merged_args.fitness_threshold = config.fitness_threshold
    merged_args.log_dir = config.log_dir
    merged_args.log_interval = config.log_interval
    merged_args.shuffle = config.shuffle
    merged_args.threshold_bot_memory = config.threshold_bot_memory
    merged_args.threshold_top_memory = config.threshold_top_memory
    merged_args.seed_memory = config.seed_memory or "configs/seed_memory.yml"
    merged_args.number_example_prompts = config.number_example_prompts
    merged_args.delta = config.delta
    merged_args.k_elites = config.k_elites
    merged_args.dataset = config.sample_prompts or "./data/do-not-answer.json"
    merged_args.config_file = args.config_file
    
    if args.num_samples is not None: merged_args.num_samples = args.num_samples
    if args.n_cells is not None: merged_args.n_cells = args.n_cells
    if args.max_iters is not None: merged_args.max_iters = args.max_iters
    if args.sim_threshold is not None: merged_args.sim_threshold = args.sim_threshold
    if args.num_mutations is not None: merged_args.num_mutations = args.num_mutations
    if args.fitness_threshold is not None: merged_args.fitness_threshold = args.fitness_threshold
    if args.log_dir is not None: merged_args.log_dir = args.log_dir
    if args.log_interval is not None: merged_args.log_interval = args.log_interval
    if args.shuffle is not None: merged_args.shuffle = args.shuffle
    if args.threshold_bot_memory is not None: merged_args.threshold_bot_memory = args.threshold_bot_memory
    if args.threshold_top_memory is not None: merged_args.threshold_top_memory = args.threshold_top_memory
    if args.seed_memory is not None: merged_args.seed_memory = args.seed_memory
    if args.number_example_prompts is not None: merged_args.number_example_prompts = args.number_example_prompts
    if args.delta is not None: merged_args.delta = args.delta
    if args.k_elites is not None: merged_args.k_elites = args.k_elites
    if args.dataset is not None: merged_args.dataset = args.dataset
    return merged_args

def load_descriptors(config):
    descriptors = {}
    for path, descriptor in zip(config.archive["path"], config.archive["descriptor"]):
        if path.endswith('.yml'):
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                if descriptor == "Persona":
                    descriptors[descriptor] = list(data['personas'].keys())
        else:
            descriptors[descriptor] = load_txt(path)
    return descriptors

def run_rainbowplus(args, config, seed_prompts=[], llms=None, fitness_fn=None, similarity_fn=None):
    if not seed_prompts:
        seed_prompts = load_json(config.sample_prompts, field="question", num_samples=args.num_samples, shuffle=args.shuffle)
    descriptors = load_descriptors(config)

    # --- 1. Initialize Archives (Legacy & Comprehensive) ---
    adv_prompts = Archive("adv_prompts")
    responses = Archive("responses")
    scores = Archive("scores")
    iters = Archive("iterations")

    all_prompts = Archive("all_prompts")
    all_responses = Archive("all_responses")
    all_scores = Archive("all_scores")
    all_similarities = Archive("all_similarities")
    rejection_reasons = Archive("rejection_reasons")
    all_queris = Archive("all_queries")
    
    # --- 2. Initialize Lineage Archives (Archive Objects) ---
    all_prompt_ids = Archive("all_prompt_ids")
    all_parent_ids = Archive("all_parent_ids")
    all_seed_ids = Archive("all_seed_ids")

    # --- 3. Initialize Growing Archive ---
    behavior_dim = 384 # Matching all-MiniLM-L6-v2 dimensions
    

    GA = GrowingArchive(
            args.n_cells, 
            behavior_dim, 
            args.fitness_threshold,
            args.delta,
            args.k_elites)
    

    attack_memory = AttackMemory(
            args.threshold_bot_memory, 
            args.threshold_top_memory)
    # --- 4. Initialize Persona Mutator ---
    persona_mutator = None
    simple_persona_mode = getattr(config, 'simple_persona_mode', False) or getattr(config, 'simple_mode', False) or config.__dict__.get('simple_persona_mode', False)
    if config.persona_config:
        selected_personas = config.archive.get('selected_personas')
        persona_type = getattr(config, 'persona_type', None) or config.__dict__.get('persona_type', 'RedTeamingExperts')
        logger.info(f"Using persona_type: {persona_type}")
        persona_mutator = PersonaMutator(config.persona_config, selected_personas=selected_personas, simple_mode=simple_persona_mode, persona_type=persona_type)

    # Log Directory
    dataset_name = Path(config.sample_prompts).stem
    if hasattr(args, 'log_dir') and args.log_dir and args.log_dir != "./logs":
        log_dir = Path(args.log_dir) / config.target_llm.model_kwargs["model"] / dataset_name
    else:
        log_dir = Path(config.log_dir) / config.target_llm.model_kwargs["model"] / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(args.seed_memory, 'r', encoding='utf-8') as f:
            seed_prompts_memory = yaml.safe_load(f) # List[Dict]
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return []
    
    # --- 5. Seed Metadata Initialization ---
    seed_metadata = []
    for idx, prompt in enumerate(seed_prompts):
        pid = f"seed_{idx}"
        seed_metadata.append({
            "prompt": prompt,
            "seed_id": pid,
            "prompt_id": pid
        })
    attack_memory.add_list(seed_prompts_memory)
    # --- MAIN LOOP ---
    for i in range(args.max_iters):
        logger.info(f"#####ITERATION: {i}")

        # --- Step 1: Selection ---
        current_parent_id = None
        current_seed_id = None
        
        if i < len(seed_prompts):
            # Phase 1: Seeding
            selection = seed_metadata[i]
            prompt = selection["prompt"]
            current_parent_id = selection["prompt_id"]
            current_seed_id = selection["seed_id"]
            
            current_descriptor = persona_mutator._get_initial_persona_from_prompt(
                prompt, llms[config.mutator_llm.model_kwargs["model"]], config.mutator_llm.sampling_params
            )
        else:
            # Phase 2: Evolution from GA
            parent_cell = GA.sample_parent()
            if parent_cell:
                prompt = parent_cell["prompt"]
                current_descriptor = parent_cell["descriptor"]
                # Inherit ID info from parent
                current_parent_id = parent_cell.get("prompt_id")
                current_seed_id = parent_cell.get("seed_id")
            else:
                # Fallback
                selection = random.choice(seed_metadata)
                prompt = selection["prompt"]
                current_parent_id = selection["prompt_id"]
                current_seed_id = selection["seed_id"]
                current_descriptor = persona_mutator._get_initial_persona_from_prompt(
                    prompt, llms[config.mutator_llm.model_kwargs["model"]], config.mutator_llm.sampling_params
                )

        # --- Step 2: Descriptor & Mutation ---
        descriptor = {}
        selected_persona = None
        number_of_persona = 1
        
        # Sample descriptors
        for key, values in descriptors.items():
            if key == "Persona":
                persona_name, persona_details = persona_mutator._generate_persona(
                    prompt, 
                    current_descriptor,
                    llms[config.mutator_llm.model_kwargs["model"]], 
                    config.mutator_llm.sampling_params,  
                    number_of_persona=number_of_persona
                )
                descriptor[key] = persona_name
                selected_persona = (persona_name, persona_details)
            else:
                descriptor[key] = random.choice(values)

        log_key = tuple(descriptor.values())

        # Format YAML string securely
        persona_name, persona_details = selected_persona
        details_to_dump = {'title': persona_name, **persona_details}
        persona_yaml_string = yaml.dump(details_to_dump, default_flow_style=False, indent=4, sort_keys=False)

        sample_top_prompts = attack_memory.get_prompts_top(args.number_example_prompts)
        sample_bot_prompts = attack_memory.get_prompts_bot(args.number_example_prompts)
        failed_yml_strings = [format_example(entry) for entry in sample_top_prompts]
        failed_example_text = "\n\nHere's another example of an unwell-structured prompt and persona:\n\n".join(failed_yml_strings)
        success_yml_strings = [format_example(entry) for entry in sample_bot_prompts]
        successful_example_text = "\n\nHere's another example of a well-structured prompt and persona:\n\n".join(success_yml_strings)

        prompt_ = MUTATOR_PROMPT.format(
            failed_examples_text=failed_example_text,
            successful_examples_text=successful_example_text,
            risk=descriptor["Category"], # Adjust based on your config keys
            style=descriptor["Style"],
            persona_yaml_details=persona_yaml_string,
            prompt=prompt
        )
        
        mutated_prompts = llms[config.mutator_llm.model_kwargs["model"]].batch_generate(
            [prompt_] * args.num_mutations, config.mutator_llm.sampling_params
        )
        original_mutated_prompts = mutated_prompts.copy()
        
        # --- Step 3: Filtering & Evaluation ---
        # Filter by similarity
        mutated_prompts = [
            p for p in mutated_prompts if similarity_fn.score(p, prompt_) < args.sim_threshold
        ]

        if mutated_prompts:
            # Responses
            target_prompts = [TARGET_PROMPT.format(prompt=p.strip()) for p in mutated_prompts]
            target_model = config.target_llm.model_kwargs["model"]
            candidates = llms[target_model].batch_generate(target_prompts, config.target_llm.sampling_params)

            # Scores
            fitness_scores = fitness_fn.batch_score(mutated_prompts, candidates, config.fitness_llm.sampling_params)
            sim_scores = [similarity_fn.score(p, prompt) for p in mutated_prompts]
            
            # Embeddings (All prompts)
            all_behaviors = persona_mutator.model.encode(mutated_prompts)

            # Update GA & Prepare Logs
            batch_prompt_ids = []
            batch_reasons = []
            
            # NOTE: We use log_key based on descriptors for archiving, not for GA logic
            log_key_tuple = (descriptor.get("Category"), descriptor.get("Style"), persona_name)

            for idx, (m_prompt, response, fit, sim, behavior) in enumerate(zip(mutated_prompts, candidates, fitness_scores, sim_scores, all_behaviors)):
                new_pid = str(uuid.uuid4())
                batch_prompt_ids.append(new_pid)
                
                status = "accepted"
                if sim > args.sim_threshold:
                    status = "similarity_too_high"
                
                if status == "accepted":
                    eval_item = {
                        "prompt": m_prompt,
                        "candidate": response,
                        "descriptor": selected_persona,
                        "fitness": fit,
                        "behavior": behavior,
                        "iter": i,
                        # Lineage info
                        "prompt_id": new_pid,
                        "parent_id": current_parent_id,
                        "seed_id": current_seed_id
                    }
                    
                    # Update GA
                    ga_result = GA.add_evaluation(eval_item)
                    candidate_memory = {
                        "behavior": behavior,
                        "prompt": m_prompt,
                        "score": fit,
                        "persoa": selected_persona
                    }
                    attack_memory.add(candidate_memory)
                    if "rejected" in ga_result:
                        status = ga_result
                    else:
                        status = ga_result # e.g. added_new_niche

                batch_reasons.append(status)

            # --- Step 4: Update Archives (using Archive Objects) ---
            def update_archive(archive_obj, k, data):
                if not archive_obj.exists(k): archive_obj.add(k, data)
                else: archive_obj.extend(k, data)

            # Update Content Archives
            update_archive(all_prompts, log_key_tuple, mutated_prompts)
            update_archive(all_responses, log_key_tuple, candidates)
            update_archive(all_scores, log_key_tuple, fitness_scores)
            update_archive(all_similarities, log_key_tuple, sim_scores)
            update_archive(rejection_reasons, log_key_tuple, batch_reasons)
            #update_archive(all_iters, log_key_tuple, [i] * len(mutated_prompts)
            
            # Update Lineage Archives (Pipeline of IDs)
            update_archive(all_prompt_ids, log_key_tuple, batch_prompt_ids)
            update_archive(all_parent_ids, log_key_tuple, [current_parent_id] * len(mutated_prompts))
            update_archive(all_seed_ids, log_key_tuple, [current_seed_id] * len(mutated_prompts))

            # Also update legacy archives for backward compatibility
            # (Optional: filter only successful ones here if needed, but let's keep simple)
            
            # --- Step 5: Logging ---
            if (i + 1) % args.log_interval == 0 or (i + 1) == args.max_iters:
                timestamp = time.strftime(r"%Y%m%d-%H%M%S")
                
                # Save GA State (Centroids)
                save_ga_iteration_log(
                    log_dir, GA, timestamp, iteration=i, max_iters=args.max_iters
                )
                
                # Save Comprehensive Log (All History + Lineage + GA State)
                
                
                

    # Final Save
    final_ts = time.strftime(r"%Y%m%d-%H%M%S")
    save_ga_iteration_log(log_dir, GA, final_ts, iteration=-1, max_iters=args.max_iters)
    
    save_attack_memory_log(log_dir, attack_memory, final_ts)
    
    save_ga_comprehensive_log(
        log_dir, GA,
        all_prompts, all_responses, all_scores, all_similarities, rejection_reasons,
        final_ts, iteration=-1,
        all_prompt_ids=all_prompt_ids,
        all_parent_ids=all_parent_ids,
        all_seed_ids=all_seed_ids,
        max_iters=args.max_iters
    )
    return adv_prompts, responses, scores # Return legacy archives to keep interface consistent if needed

if __name__ == "__main__":
    args = parse_arguments()
    config = ConfigurationLoader.load(args.config_file)
    merged_args = merge_config_with_args(config, args)
    if not config.sample_prompts:
        config.sample_prompts = merged_args.dataset

    llms = initialize_language_models(config)
    fitness_fn = LlamaGuard(config.fitness_llm)
    similarity_fn = BleuScoreNLTK()

    print(config)

    run_rainbowplus(
        merged_args,
        config,
        seed_prompts=[],
        llms=llms,
        fitness_fn=fitness_fn,
        similarity_fn=similarity_fn,
    )