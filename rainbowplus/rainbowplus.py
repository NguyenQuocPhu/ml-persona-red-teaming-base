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
from pathlib import Path
import yaml
import numpy as np

# --- Project-Specific Imports ---
from rainbowplus.scores import BleuScoreNLTK, OpenAIGuard, LlamaGuard
from rainbowplus.utils import (
    load_txt,
    load_json,
    initialize_language_models,
    save_iteration_log,
    save_comprehensive_log,
)
from rainbowplus.archive import Archive
from rainbowplus.configs import ConfigurationLoader
from rainbowplus.prompts import MUTATOR_PROMPT, TARGET_PROMPT
from rainbowplus.configs.base import LLMConfig
from rainbowplus.mutators.persona import PersonaMutator  # Key component for persona-based mutation

# --- Setup ---

# Set fixed random seed for reproducibility
RANDOM_SEED = 15
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# --- QD Algorithm Utilities ---

import numpy as np
from typing import Dict
import random
from copy import deepcopy

def cosine_distance_batch(X, Y):
    """
    Calculates the batch-wise cosine distance between two sets of vectors.
    Distance = 1 - Cosine Similarity
    
    Args:
        X: A (n_vectors_X, n_dims) numpy array.
        Y: A (n_vectors_Y, n_dims) numpy array.
        
    Returns:
        A (n_vectors_X, n_vectors_Y) array of cosine distances.
    """
    # 1 - (X . Y) / (||X|| * ||Y||)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    
    # Avoid division by zero for zero-vectors
    X_norm[X_norm == 0] = 1e-12
    Y_norm[Y_norm == 0] = 1e-12
    
    dot_product = X @ Y.T
    similarity = dot_product / (X_norm @ Y_norm.T)
    
    # Clip for numerical stability (floating point errors)
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return 1.0 - similarity

def _compute_distance_to_centroids(b_vector, centroids):
    """
    Finds the nearest centroid for a given behavior vector.
    
    Args:
        b_vector: The (n_dims,) behavior vector of the new solution.
        centroids: The (n_centroids, n_dims) array of existing centroids.
        
    Returns:
        A tuple of (min_distance, nearest_centroid_id).
    """
    if len(centroids) == 0:
        return np.inf, -1
    
    # Ensure b_vector is a 2D array for batch calculation
    if b_vector.ndim == 1:
        b_vector = b_vector.reshape(1, -1)
        
    distances = cosine_distance_batch(b_vector, centroids)
    c_id = np.argmin(distances)
    return distances[0, c_id], c_id

class GrowingArchive:
    """
    Implements a Quality-Diversity (QD) archive based on the 'Growing Archive' concept.
    
    Instead of a fixed grid (like MAP-Elites), this archive dynamically adds and
    prunes centroids based on novelty and a minimum distance threshold (dmin).
    This is well-suited for high-dimensional or continuous behavior spaces.
    """
    def __init__(self, n_cells: int, n_behavior_dim: int, fitness_threshold: float):
        """
        Initializes the archive structures.
        
        Args:
            n_cells: The maximum number of centroids (elites) the archive can hold.
            n_behavior_dim: The dimensionality of the behavior vectors.
            fitness_threshold: The minimum fitness required to be considered for the archive.
        """
        self.n_cells = n_cells
        self.fitness_threshold = fitness_threshold

        # `centroids` stores the behavior vectors of the elites
        self.centroids = np.empty((n_cells, n_behavior_dim), dtype = np.float32)
        # `elites` stores the full evaluation data (prompt, score, etc.) mapped by cell_id
        self.elites : Dict[int, Dict] = {}
        self.elites_backup : Dict[int, Dict] = {} # Used during repair operations
        self.n_centroids = 0 # Current number of active centroids
        self.dmin = np.inf # Novelty threshold: minimum distance between any two centroids

    def _compute_dmin(self):
        """
        Calculates `dmin`, the minimum distance between any two active centroids.
        This value is the threshold for novelty-based pruning.
        It also pre-computes neighbor distances for efficient pruning.
        """
        if self.n_centroids < 2:
            self.dmin = np.inf
            return
            
        active_centroids = self.centroids[:self.n_centroids]
        distances = cosine_distance_batch(active_centroids, active_centroids)

        np.fill_diagonal(distances, np.inf) # Ignore distance to self
        self.dmin = np.min(distances)

        # Store sorted neighbors for each centroid (used in pruning)
        self.c_id_neighbors = np.argsort(distances, axis = 1)
        self.d_neighbors = np.array([distances[i][self.c_id_neighbors[i]] for i in range(self.n_centroids)])

    def _set_new_elite(self, cell_id: int, evaluation: Dict, is_backup: bool = True):
        """Helper function to update or add a new elite solution to the archive."""
        self.elites[cell_id] = deepcopy(evaluation)
        if is_backup:
            # Backup is for `__apply_repair`
            self.elites_backup = deepcopy(evaluation)
            
    def __apply_repair(self, pruned_cell_id: int):
        """
        Re-evaluates and re-assigns elites to the nearest centroid after a pruning operation.
        This ensures archive consistency when a centroid is removed or moved.
        """
        active_centroids = self.centroids[:self.n_centroids]
        keys_to_check = list(self.elites.keys())
        if keys_to_check in active_centroids:
            keys_to_check.remove(pruned_cell_id)
        if not keys_to_check:
            return

        behaviors_to_check = np.array([self.elites[k]["behavior"]] for k in keys_to_check)
        distances = cosine_distance_batch(behaviors_to_check, active_centroids)
        new_cell_ids = np.argmin(distances, axis = 1)

        # Check if any elite needs to be re-assigned
        for i, old_cell_id in enumerate(keys_to_check):
            new_cell_id = new_cell_ids[i]
            if new_cell_id != old_cell_id:
                # Restore from backup, will be re-evaluated in the next iteration
                self.elites[old_cell_id] = deepcopy(self.elites_backup[old_cell_id])

    def add_evaluation(self, evaluation: Dict):
        """
        Adds a new evaluated solution to the archive based on fitness and novelty.
        
        Args:
            evaluation: A dictionary containing at least {"fitness": float, "behavior": np.array}
        """
        # 1. Discard solutions that do not meet the minimum fitness
        if evaluation["fitness"] < self.fitness_threshold:
            return
            
        b_vector = evaluation["behavior"].reshape(1, -1)
        active_centroids = self.centroids[:self.n_centroids]

        # 2. If the archive is not full, add the new solution as a new centroid.
        if self.n_centroids < self.n_cells:
            new_cell_id = self.n_centroids
            self._set_new_elite(new_cell_id, evaluation, is_backup = True)
            self.n_centroids += 1
            
            # If archive just became full, calculate dmin for the first time
            if self.n_centroids == self.n_cells:
                self._compute_dmin()
            return
        
        # 3. If archive is full, find the nearest existing centroid.
        d_to_nearest, cell_id = _compute_distance_to_centroids(b_vector, active_centroids)

        # 4. If the new solution is more novel than `dmin`, it replaces an existing centroid.
        if d_to_nearest > self.dmin:
            # --- Pruning Strategy ---
            # Find the two closest centroids (A and B) in the entire archive.
            centroid_A = np.argmin(self.d_neighbors[:, 0])
            centroid_B = self.c_id_neighbors[centroid_A, 0]

            # Find their *second* nearest neighbors.
            dist_A_to_neighbor_2 = self.d_neighbors[centroid_A, 1]
            dist_B_to_neighbor_2 = self.d_neighbors[centroid_B, 1]

            # Prune the centroid (A or B) that is in the more "crowded" area
            # (i.e., the one whose second neighbor is closer).
            if dist_A_to_neighbor_2 < dist_B_to_neighbor_2:
                pruned_cell_id = centroid_A
            else:
                pruned_cell_id = centroid_B
                
            # Replace the pruned centroid with the new, novel solution
            self.centroids[pruned_cell_id] = b_vector
            self._set_new_elite(pruned_cell_id, evaluation, is_backup = True)
            
            # Recalculate dmin and repair the archive
            self._compute_dmin()
            self.__apply_repair(pruned_cell_id)
            return
        
        # 5. If the solution is not novel, check if it improves the fitness of its nearest centroid.
        if evaluation["fitness"] > self.elites[cell_id]["fitness"]:
            self._set_new_elite(cell_id, evaluation, is_backup = False) # No backup, not a structural change

    def sample_parent(self):
        """Selects a random elite from the archive to be a 'parent' for mutation."""
        if not self.elites:
            return None
        
        random_key = random.choice(list(self.elites.keys()))
        return self.elites[random_key]

# --- Main Script Setup ---

def parse_arguments():
    """
    Parse command-line arguments for adversarial prompt generation.
    """
    parser = argparse.ArgumentParser(description="Adversarial Prompt Generation")
    # ... (Argument definitions) ...
    parser.add_argument(
        "--num_samples", type=int, help="Number of initial seed prompts"
    )
    parser.add_argument(
        "--n_cells", type=int, default = 100, help="Number of cells in the archive"
    )
    parser.add_argument(
        "--max_iters", type=int, help="Maximum number of iteration steps"
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        help="Similarity threshold for prompt mutation",
    )
    parser.add_argument(
        "--num_mutations",
        type=int,
        help="Number of prompt mutations per iteration",
    )
    parser.add_argument(
        "--fitness_threshold",
        type=float,
        help="Minimum fitness score to add prompt to archive",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="./configs/base-opensource-persona.yml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log_dir", type=str, help="Directory for storing logs"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="Number of iterations between log saves",
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset name"
    )
    parser.add_argument(
        "--target_llm",
        type=str,
        help="Path to repository of target LLM",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle seed prompts"
    )
    return parser.parse_args()


def merge_config_with_args(config, args):
    """
    Merge configuration values with command-line arguments.
    Command-line arguments take precedence when explicitly provided.
    """
    # Create a new args object with config defaults
    merged_args = type('Args', (), {})()
    
    # Set values from config first
    merged_args.num_samples = config.num_samples
    merged_args.max_iters = config.max_iters
    merged_args.sim_threshold = config.sim_threshold
    merged_args.num_mutations = config.num_mutations
    merged_args.fitness_threshold = config.fitness_threshold
    merged_args.log_dir = config.log_dir
    merged_args.log_interval = config.log_interval
    merged_args.shuffle = config.shuffle
    merged_args.dataset = config.sample_prompts or "./data/do-not-answer.json"
    merged_args.config_file = args.config_file
    
    # Override with command-line arguments if provided
    if args.num_samples is not None:
        merged_args.num_samples = args.num_samples
    if args.n_cells is not None:
        merged_args.n_cells = args.n_cells
    if args.max_iters is not None:
        merged_args.max_iters = args.max_iters
    if args.sim_threshold is not None:
        merged_args.sim_threshold = args.sim_threshold
    if args.num_mutations is not None:
        merged_args.num_mutations = args.num_mutations
    if args.fitness_threshold is not None:
        merged_args.fitness_threshold = args.fitness_threshold
    if args.log_dir is not None:
        merged_args.log_dir = args.log_dir
    if args.log_interval is not None:
        merged_args.log_interval = args.log_interval
    if args.shuffle is not None:
        merged_args.shuffle = args.shuffle
    if args.dataset is not None:
        merged_args.dataset = args.dataset
    
    return merged_args


def load_descriptors(config):
    """
    Load descriptors (Risk Category, Attack Style, Persona) from specified paths.
    """
    descriptors = {}
    for path, descriptor in zip(config.archive["path"], config.archive["descriptor"]):
        if path.endswith('.yml'):
            # Load personas from YAML config
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                # For personas, we use the persona keys (names) as the descriptors
                if descriptor == "Persona":
                    descriptors[descriptor] = list(data['personas'].keys())
        else:
            # Load other descriptors (categories and styles) from simple .txt files
            descriptors[descriptor] = load_txt(path)
    return descriptors


# --- Main Execution ---

def run_rainbowplus(
    args, config, seed_prompts=[], llms=None, fitness_fn=None, similarity_fn=None, model_embedding = None
):
    """
    Main execution loop for the QD-driven adversarial prompt generation.
    
    This function integrates:
    1. A `GrowingArchive` (QD algorithm) to store and select diverse, high-fitness solutions.
    2. A `PersonaMutator` to generate context-aware and persona-driven mutations.
    """
    # Load seed prompts if not provided
    if not seed_prompts:
        seed_prompts = load_json(
            config.sample_prompts,
            field="question",
            num_samples=args.num_samples,
            shuffle=args.shuffle,
        )

    # Load category descriptors (Risk, Style, Persona)
    descriptors = load_descriptors(config)

    # Initialize archives for storing successful adversarial prompts
    adv_prompts = Archive("adv_prompts")
    responses = Archive("responses")
    scores = Archive("scores")
    iters = Archive("iterations")

    # Initialize comprehensive archives for logging *all* generated prompts (for analysis)
    all_prompts = Archive("all_prompts")
    all_responses = Archive("all_responses")
    all_scores = Archive("all_scores")
    all_similarities = Archive("all_similarities")
    rejection_reasons = Archive("rejection_reasons")
    
    # Define the dimensionality of the behavior space
    # In this setup, the behavior is the semantic embedding of the prompt
    behavior_dim = 768  # Assuming base-model embedding size (e.g., MiniLM)
    
    # Initialize the Quality-Diversity archive
    GA = GrowingArchive(
        args.n_cells,
        behavior_dim,
        args.fitness_threshold
    )
    
    # Initialize persona mutator
    persona_mutator = None
    simple_persona_mode = getattr(config, 'simple_persona_mode', False) or getattr(config, 'simple_mode', False) or config.__dict__.get('simple_persona_mode', False)
    if config.persona_config:
        selected_personas = config.archive.get('selected_personas')
        persona_type = getattr(config, 'persona_type', None) or config.__dict__.get('persona_type', 'RedTeamingExperts')
        logger.info(f"Using persona_type: {persona_type}")
        
        # This object handles persona generation and formatting
        persona_mutator = PersonaMutator(config.persona_config, selected_personas=selected_personas, simple_mode=simple_persona_mode, persona_type=persona_type)

    # Prepare log directory
    dataset_name = Path(config.sample_prompts).stem
    if hasattr(args, 'log_dir') and args.log_dir and args.log_dir != "./logs":
        log_dir = Path(args.log_dir) / config.target_llm.model_kwargs["model"] / dataset_name
    else:
        log_dir = Path(config.log_dir) / config.target_llm.model_kwargs["model"] / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # --- Main Adversarial Generation Loop ---
    for i in range(args.max_iters):
        logger.info(f"#####ITERATION: {i}")

        # --- 1. Parent Selection ---
        if i < len(seed_prompts):
            # Phase 1: Seeding. Use initial prompts from the dataset.
            prompt = seed_prompts[i]
            # Get an initial persona for this seed prompt
            current_descriptor = persona_mutator._get_initial_persona_from_prompt(
                prompt, 
                llms[config.mutator_llm.model_kwargs["model"]], 
                config.mutator_llm.sampling_params,
            )
        else:
            # Phase 2: Evolution. Sample a parent from the archive.
            adv_value = GA.n_centroids # Corrected: Access attribute
            if adv_value:
                # Sample a high-performing, diverse parent
                cell = GA.sample_parent()
                prompt = cell["prompt"]
                current_descriptor = cell["descriptor"] # Use the persona of the parent
            else:
                # Fallback if archive is empty (e.g., fitness_threshold is too high)
                prompt = random.choice(seed_prompts)
                current_descriptor = persona_mutator._get_initial_persona_from_prompt(
                    prompt, 
                    llms[config.mutator_llm.model_kwargs["model"]], 
                    config.mutator_llm.sampling_params,
                )

        # --- 2. Descriptor Sampling ---
        descriptor = {}
        selected_persona = None
        number_of_persona = 5 # Config for persona generation algorithm
        
        # Sample descriptors for mutation
        for key, values in descriptors.items():
            if key == "Persona":
                # For Persona: Use the mutator's intelligent generation
                # This finds/creates a persona that fits the parent prompt
                persona_name, persona_details = persona_mutator._generate_contextual_persona(
                    prompt, 
                    current_descriptor,
                    llms[config.mutator_llm.model_kwargs["model"]], 
                    config.mutator_llm.sampling_params,  
                    number_of_persona=number_of_persona
                )
                descriptor[key] = persona_name
                selected_persona = (persona_name, persona_details)
            else:
                # For other descriptors (Risk, Style): Use random selection
                descriptor[key] = random.choice(values)

        # Create unique key for this combination of descriptors
        key = tuple(descriptor.values())

        # 1. Lấy (name, details)
        persona_name, persona_details = selected_persona
        # 2. Tái cấu trúc dict để dump, thêm 'title' vào lại
        details_to_dump = {'title': persona_name, **persona_details}
        
        # 3. Dùng yaml.dump để tạo chuỗi YAML chuẩn
        persona_yaml_string = yaml.dump(
            details_to_dump, 
            default_flow_style=False, 
            indent=4, 
            sort_keys=False
        )
        # --- KẾT THÚC SỬA LỖI ---

        mutator_model = config.mutator_llm.model_kwargs["model"]
        
        # Use default mutator
        prompt_ = MUTATOR_PROMPT.format(
            risk = descriptor["Risk Category"],
            style = descriptor["Attack Style"],
            # 4. Truyền đúng chuỗi YAML vào đây
            persona_yaml_details = persona_yaml_string, 
            prompt=prompt
        )
        
        # Generate a batch of new adversarial prompts
        mutated_prompts = llms[mutator_model].batch_generate(
            [prompt_] * args.num_mutations, config.mutator_llm.sampling_params
        )

        # Store all original mutated prompts before filtering (for comprehensive logging)
        original_mutated_prompts = mutated_prompts.copy()

        # --- 4. Filtering (Similarity) ---
        # Filter 1: Remove new prompts that are too similar to the parent
        mutated_prompts = [
            p
            for p in mutated_prompts
            if similarity_fn.score(p, prompt_) < args.sim_threshold
            ######## BỔ SUNG THÊM ĐIỀU KIỆN FILTER (Add more filter conditions here if needed)
        ]

        # --- 5. Evaluation ---
        if mutated_prompts:
            # Evaluate prompts: Get responses from the Target LLM
            target_prompts = [
                TARGET_PROMPT.format(prompt=p.strip()) for p in mutated_prompts
            ]
            target_model = config.target_llm.model_kwargs["model"]
            candidates = llms[target_model].batch_generate(
                target_prompts, config.target_llm.sampling_params
            )

            # Evaluate fitness: Score the responses (e.g., check for refusal)
            fitness_scores = fitness_fn.batch_score(
                mutated_prompts, candidates, config.fitness_llm.sampling_params
            )

            # Filter 2: Keep only prompts that meet the fitness threshold
            filtered_data = [
                (p, c, s)
                for p, c, s in zip(mutated_prompts, candidates, fitness_scores)
                if s > args.fitness_threshold
            ]

            # --- 6. Archive Addition ---
            if filtered_data:
                # Corrected Logic: Unzip first, then process
                
                # 1. Unpack filtered data
                filtered_prompts, filtered_candidates, filtered_scores = zip(
                    *filtered_data
                )

                # 2. Calculate the 'behavior' vector for each new prompt.
                # Here, the behavior is the semantic embedding of the prompt itself.
                all_behaviors = persona_mutator.model.encode(filtered_prompts)

                # 3. Iterate and add each new successful solution to the QD archive individually
                for idx in range(len(filtered_prompts)):
                    evaluation_item = {
                        "prompt": filtered_prompts[idx],
                        "candidate": filtered_candidates[idx],
                        "descriptor": selected_persona, # Store the persona used
                        "fitness": filtered_scores[idx],
                        "behavior": all_behaviors[idx]  # The semantic vector
                    }
                    # The archive will internally decide whether to add, improve, or discard
                    GA.add_evaluation(evaluation_item)

                # 4. Log successful mutations
                logger.info(f"Prompt for Mutator: {prompt_}")
                logger.info(f"Mutated Prompt: {filtered_prompts}")
                logger.info(f"Candidate: {filtered_candidates}")
                logger.info(f"Score: {filtered_scores}")
                logger.info("\n\n\n")

                # 5. Update legacy archives (for standard logging)
                if not adv_prompts.exists(key):
                    adv_prompts.add(key, filtered_prompts)
                    responses.add(key, filtered_candidates)
                    scores.add(key, filtered_scores)
                    iters.add(key, [i] * len(filtered_prompts))
                else:
                    adv_prompts.extend(key, filtered_prompts)
                    responses.extend(key, filtered_candidates)
                    scores.extend(key, filtered_scores)
                    iters.extend(key, [i] * len(filtered_prompts))

        # --- 7. Comprehensive Logging (for ALL attempts) ---
        # This logs *all* generated prompts (even failures) for detailed analysis
        all_responses_for_key = []
        all_scores_for_key = []
        all_similarities_for_key = []
        rejection_reasons_for_key = []

        if original_mutated_prompts:
            # Get responses for ALL original prompts
            all_target_prompts = [
                TARGET_PROMPT.format(prompt=p.strip()) for p in original_mutated_prompts
            ]
            target_model = config.target_llm.model_kwargs["model"]
            all_candidates = llms[target_model].batch_generate(
                all_target_prompts, config.target_llm.sampling_params
            )

            # Score fitness of ALL prompts
            all_fitness_scores = fitness_fn.batch_score(
                original_mutated_prompts, all_candidates, config.fitness_llm.sampling_params
            )

            # Calculate similarity scores for ALL prompts
            all_sim_scores = [similarity_fn.score(p, prompt_) for p in original_mutated_prompts]

            # Determine rejection reasons for each prompt
            for j, (prompt, response, score, sim_score) in enumerate(zip(original_mutated_prompts, all_candidates, all_fitness_scores, all_sim_scores)):
                all_responses_for_key.append(response)
                all_scores_for_key.append(score)
                all_similarities_for_key.append(sim_score)
                
                # Record *why* a prompt was rejected (or accepted)
                if sim_score >= args.sim_threshold:
                    rejection_reasons_for_key.append("similarity_too_high")
                elif score <= args.fitness_threshold:
                    rejection_reasons_for_key.append("fitness_too_low")
                else:
                    rejection_reasons_for_key.append("accepted")

            # Update comprehensive archives
            if not all_prompts.exists(key):
                all_prompts.add(key, original_mutated_prompts)
                all_responses.add(key, all_responses_for_key)
                all_scores.add(key, all_scores_for_key)
                all_similarities.add(key, all_similarities_for_key)
                rejection_reasons.add(key, rejection_reasons_for_key)
            else:
                all_prompts.extend(key, original_mutated_prompts)
                all_responses.extend(key, all_responses_for_key)
                all_scores.extend(key, all_scores_for_key)
                all_similarities.extend(key, all_similarities_for_key)
                rejection_reasons.extend(key, rejection_reasons_for_key)

        # --- 8. Periodic Saving ---
        # Global saving (overwrites the 'global' log every iteration)
        save_iteration_log(
            log_dir, adv_prompts, responses, scores, iters, "global", iteration=-1, max_iters=args.max_iters
        )
        save_comprehensive_log(
            log_dir, all_prompts, all_responses, all_scores, all_similarities, rejection_reasons, "global", iteration=-1,
            max_iters=args.max_iters
        )

        # Periodic logging (creates timestamped snapshot)
        if i > 0 and (i + 1) % args.log_interval == 0:
            timestamp = time.strftime(r"%Y%m%d-%H%M%S")
            save_iteration_log(
                log_dir, adv_prompts, responses, scores, iters, timestamp, iteration=i, max_iters=args.max_iters
            )
            save_comprehensive_log(
                log_dir, all_prompts, all_responses, all_scores, all_similarities, rejection_reasons, timestamp, iteration=i,
                max_iters=args.max_iters
            )

    # --- End of Loop ---
    
    # Save final log snapshot
    timestamp = time.strftime(r"%Y%m%d-%H%M%S")
    save_iteration_log(log_dir, adv_prompts, responses, scores, iters, timestamp, iteration=i, max_iters=args.max_iters)
    save_comprehensive_log(
        log_dir, all_prompts, all_responses, all_scores, all_similarities, rejection_reasons, timestamp, iteration=i,
        max_iters=args.max_iters
    )

    # Return final archives
    return adv_prompts, responses, scores


if __name__ == "__main__":
    # --- Script Entry Point ---
    
    # 1. Parse command-line arguments
    args = parse_arguments()

    # 2. Load configuration file
    config = ConfigurationLoader.load(args.config_file)

    # 3. Merge config with command-line args (args override config)
    merged_args = merge_config_with_args(config, args)

    # Update configuration
    if not config.sample_prompts:
        config.sample_prompts = merged_args.dataset

    # 4. Initialize all required LLMs (Mutator, Target, Fitness)
    llms = initialize_language_models(config)
    
    # 5. Set the fitness function (e.g., a guard model)
    fitness_fn = LlamaGuard(config.fitness_llm.model_kwargs)
    
    # 6. Set the similarity function (e.g., BLEU score)
    similarity_fn = BleuScoreNLTK()

    # Show final configuration
    print(config)

    # 7. Start the main generation process
    run_rainbowplus(
        merged_args,
        config,
        seed_prompts=[],
        llms=llms,
        fitness_fn=fitness_fn,
        similarity_fn=similarity_fn,
    )