
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
from rainbowplus.mutators.persona import PersonaMutator

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


def parse_arguments():
    """
    Parse command-line arguments for adversarial prompt generation.

    Returns:
        Parsed arguments with configuration for the script
    """
    parser = argparse.ArgumentParser(description="Adversarial Prompt Generation")
    parser.add_argument(
        "--num_samples", type=int, help="Number of initial seed prompts"
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
        default="./configs/base.yml",
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
    Load descriptors from specified paths.

    Args:
        config: Configuration object with archive paths

    Returns:
        Dictionary of descriptors loaded from text files
    """
    descriptors = {}
    for path, descriptor in zip(config.archive["path"], config.archive["descriptor"]):
        if path.endswith('.yml'):
            # Load personas from YAML
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                # For personas, we want to use the persona keys
                if descriptor == "Persona":
                    # Check if specific personas are selected
                    if 'selected_personas' in config.archive:
                        # Only include selected personas
                        descriptors[descriptor] = config.archive['selected_personas']
                    else:
                        # Include all personas if none specified
                        descriptors[descriptor] = list(data['personas'].keys())
                else:
                    descriptors[descriptor] = list(data['personas'].keys())
        else:
            # Always load other descriptors (categories and styles)
            descriptors[descriptor] = load_txt(path)
    return descriptors


def run_rainbowplus(
    args, config, seed_prompts=[], llms=None, fitness_fn=None, similarity_fn=None
):
    """
    Main function to execute adversarial prompt generation process.
    Handles prompt mutation, model interactions, and logging.
    """
    # Load seed prompts
    if not seed_prompts:
        seed_prompts = load_json(
            config.sample_prompts,
            field="question",
            num_samples=args.num_samples,
            shuffle=args.shuffle,
        )

    # Load category descriptors
    descriptors = load_descriptors(config)

    # Initialize archives for adversarial prompts
    adv_prompts = Archive("adv_prompts")
    responses = Archive("responses")
    scores = Archive("scores")
    iters = Archive("iterations")

    # Initialize comprehensive archives for all generated prompts
    all_prompts = Archive("all_prompts")
    all_responses = Archive("all_responses")
    all_scores = Archive("all_scores")
    all_similarities = Archive("all_similarities")
    rejection_reasons = Archive("rejection_reasons")
    # New: Track prompt lineage
    all_prompt_ids = Archive("all_prompt_ids")
    all_parent_ids = Archive("all_parent_ids")
    all_seed_ids = Archive("all_seed_ids")

    # ID counters
    prompt_id_counter = 0
    prompt_id_map = {}  # prompt text -> prompt_id (for parent lookup)
    seed_id_map = {}    # prompt text -> seed_id

    # Assign IDs to seed prompts
    for idx, seed in enumerate(seed_prompts):
        pid = f"seed_{idx}"
        prompt_id_map[seed] = pid
        seed_id_map[seed] = pid

    # Initialize persona mutator only if needed
    persona_mutator = None
    simple_persona_mode = getattr(config, 'simple_persona_mode', False) or getattr(config, 'simple_mode', False) or config.__dict__.get('simple_persona_mode', False)
    if config.mutation_strategy in ["persona", "combined", "combined-fit"] and config.persona_config:
        selected_personas = config.archive.get('selected_personas')
        #persona_type = getattr(config, 'persona_type', 'RedTeamingExperts')  # Default to RedTeamingExperts
        # Read persona_type more robustly
        persona_type = getattr(config, 'persona_type', None) or config.__dict__.get('persona_type', 'RedTeamingExperts')
        logger.info(f"Using persona_type: {persona_type}")
        #print(f"DEBUG: PersonaMutator initialized with persona_type = {persona_type}")
        persona_mutator = PersonaMutator(config.persona_config, selected_personas=selected_personas, simple_mode=simple_persona_mode, persona_type=persona_type)

    # Prepare log directory
    dataset_name = Path(config.sample_prompts).stem
    if hasattr(args, 'log_dir') and args.log_dir and args.log_dir != "./logs":
        log_dir = Path(args.log_dir) / config.target_llm.model_kwargs["model"] / dataset_name
    else:
        log_dir = Path(config.log_dir) / config.target_llm.model_kwargs["model"] / dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Main adversarial prompt generation loop
    for i in range(args.max_iters):
        logger.info(f"#####ITERATION: {i}")

        # Select prompt (initial seed or from existing adversarial prompts)
        if i < len(seed_prompts):
            prompt = seed_prompts[i]
            parent_id = None
            seed_id = seed_id_map[prompt]
            prompt_id = prompt_id_map[prompt]
        else:
            adv_values = adv_prompts.flatten_values()
            if adv_values:
                prompt = random.choice(adv_values)
                parent_id = prompt_id_map.get(prompt, None)
                seed_id = seed_id_map.get(prompt, None)
                prompt_id = None  # Will be assigned to new mutations
            else:
                prompt = random.choice(seed_prompts)
                parent_id = None
                seed_id = seed_id_map[prompt]
                prompt_id = prompt_id_map[prompt]

        # Sample descriptors based on strategy
        descriptor = {}
        selected_persona = None
        if config.mutation_strategy == "combined-fit" and persona_mutator:
            # Use all descriptors, but for persona use fitting
            for key, values in descriptors.items():
                if key == "Persona":
                    # Use persona mutator to find a fitting persona
                    persona_name, persona_details = persona_mutator._find_fitting_persona(
                        prompt, 
                        llms[config.mutator_llm.model_kwargs["model"]], 
                        config.mutator_llm.sampling_params
                    )
                    descriptor[key] = persona_name
                    selected_persona = (persona_name, persona_details)
                else:
                    # For other descriptor types, use random selection
                    descriptor[key] = random.choice(values)
        elif config.mutation_strategy == "combined" and persona_mutator:
            # Use all descriptors together
            for key, values in descriptors.items():
                if key == "Persona":
                    # If personas are selected, use only those
                    if 'selected_personas' in config.archive:
                        # Get a random persona from selected personas
                        persona_name = random.choice(config.archive['selected_personas'])
                        persona_details = persona_mutator.personas[persona_name]
                        descriptor[key] = persona_name
                        selected_persona = (persona_name, persona_details)
                    else:
                        # Otherwise use any persona
                        persona_name = random.choice(values)
                        descriptor[key] = persona_name
                        selected_persona = (persona_name, persona_mutator.personas[persona_name])
                else:
                    # For other descriptor types, use random selection
                    descriptor[key] = random.choice(values)
        elif config.mutation_strategy == "persona" and persona_mutator:
            # If specific personas are selected, use them directly
            if 'selected_personas' in config.archive:
                # Get a random persona from selected personas
                persona_name = random.choice(config.archive['selected_personas'])
                persona_details = persona_mutator.personas[persona_name]
                descriptor["Persona"] = persona_name
                selected_persona = (persona_name, persona_details)
            else:
                # Use persona mutator to find or generate a fitting persona
                persona_name, persona_details = persona_mutator._find_fitting_persona(
                    prompt, 
                    llms[config.mutator_llm.model_kwargs["model"]], 
                    config.mutator_llm.sampling_params
                )
                descriptor["Persona"] = persona_name
                selected_persona = (persona_name, persona_details)
        else:
            # Use random selection for other descriptors (rainbow-teaming-only)
            for key, values in descriptors.items():
                descriptor[key] = random.choice(values)

        # Create unique key for this descriptor set
        key = tuple(descriptor.values())

        # Prepare descriptor string for prompt mutation
        descriptor_str = "- " + "- ".join(
            [f"{key}: {value}\n" for key, value in descriptor.items()]
        )

        # Mutate prompts using appropriate strategy
        mutator_model = config.mutator_llm.model_kwargs["model"]
        if config.mutation_strategy == "persona" and persona_mutator:
            # Use persona mutator with pre-selected persona and num_mutations
            mutated_prompts = persona_mutator.mutate(
                prompt, 
                llms[mutator_model], 
                config.mutator_llm.sampling_params,
                selected_persona=selected_persona,
                num_mutations=args.num_mutations
            )
            # Create prompt_ for similarity check
            prompt_ = prompt
        else:
            # Use default mutator
            prompt_ = MUTATOR_PROMPT.format(
                descriptor=descriptor_str.strip(), prompt=prompt
            )
            mutated_prompts = llms[mutator_model].batch_generate(
                [prompt_] * args.num_mutations, config.mutator_llm.sampling_params
            )

        # Store all original mutated prompts before filtering
        original_mutated_prompts = mutated_prompts.copy()

        # Filter mutated prompts by similarity
        mutated_prompts = [
            p
            for p in mutated_prompts
            if similarity_fn.score(p, prompt_) < args.sim_threshold
        ]

        # Assign IDs to mutated prompts
        mutated_prompt_ids = []
        mutated_parent_ids = []
        mutated_seed_ids = []
        for mp in original_mutated_prompts:
            if i < len(seed_prompts):
                # Seed prompt, already has ID
                mpid = prompt_id_map[mp] if mp in prompt_id_map else f"seed_{i}"
                parid = None
                sid = seed_id_map[mp] if mp in seed_id_map else mpid
            else:
                mpid = f"mut_{prompt_id_counter}"
                prompt_id_counter += 1
                parid = prompt_id if prompt_id else parent_id
                sid = seed_id
            prompt_id_map[mp] = mpid
            seed_id_map[mp] = sid
            mutated_prompt_ids.append(mpid)
            mutated_parent_ids.append(parid)
            mutated_seed_ids.append(sid)

        if mutated_prompts:
            # Generate responses for mutated prompts
            target_prompts = [
                TARGET_PROMPT.format(prompt=p.strip()) for p in mutated_prompts
            ]
            target_model = config.target_llm.model_kwargs["model"]
            candidates = llms[target_model].batch_generate(
                target_prompts, config.target_llm.sampling_params
            )

            # Score fitness of mutated prompts
            fitness_scores = fitness_fn.batch_score(
                mutated_prompts, candidates, config.fitness_llm.sampling_params
            )

            # Filter prompts based on fitness threshold
            filtered_data = [
                (p, c, s)
                for p, c, s in zip(mutated_prompts, candidates, fitness_scores)
                if s > args.fitness_threshold
            ]

            if filtered_data:
                # Unpack filtered data
                filtered_prompts, filtered_candidates, filtered_scores = zip(
                    *filtered_data
                )

                # Use logger
                logger.info(f"Prompt for Mutator: {prompt_}")
                logger.info(f"Mutated Prompt: {filtered_prompts}")
                logger.info(f"Candidate: {filtered_candidates}")
                logger.info(f"Score: {filtered_scores}")
                logger.info("\n\n\n")

                # Update archives
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

        # Track all generated prompts and their rejection reasons
        all_responses_for_key = []
        all_scores_for_key = []
        all_similarities_for_key = []
        rejection_reasons_for_key = []

        # Generate responses for ALL original prompts (not just filtered ones)
        if original_mutated_prompts:
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
                
                # Determine rejection reason
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
                all_prompt_ids.add(key, mutated_prompt_ids)
                all_parent_ids.add(key, mutated_parent_ids)
                all_seed_ids.add(key, mutated_seed_ids)
            else:
                all_prompts.extend(key, original_mutated_prompts)
                all_responses.extend(key, all_responses_for_key)
                all_scores.extend(key, all_scores_for_key)
                all_similarities.extend(key, all_similarities_for_key)
                rejection_reasons.extend(key, rejection_reasons_for_key)
                all_prompt_ids.extend(key, mutated_prompt_ids)
                all_parent_ids.extend(key, mutated_parent_ids)
                all_seed_ids.extend(key, mutated_seed_ids)

        # Global saving
        save_iteration_log(
            log_dir, adv_prompts, responses, scores, iters, "global", iteration=-1, max_iters=args.max_iters
        )

        # Global comprehensive saving
        save_comprehensive_log(
            log_dir, all_prompts, all_responses, all_scores, all_similarities, rejection_reasons, "global", iteration=-1,
            all_prompt_ids=all_prompt_ids, all_parent_ids=all_parent_ids, all_seed_ids=all_seed_ids, max_iters=args.max_iters
        )

        # Periodic logging
        if i > 0 and (i + 1) % args.log_interval == 0:
            timestamp = time.strftime(r"%Y%m%d-%H%M%S")
            save_iteration_log(
                log_dir, adv_prompts, responses, scores, iters, timestamp, iteration=i, max_iters=args.max_iters
            )
            # Periodic comprehensive logging
            save_comprehensive_log(
                log_dir, all_prompts, all_responses, all_scores, all_similarities, rejection_reasons, timestamp, iteration=i,
                all_prompt_ids=all_prompt_ids, all_parent_ids=all_parent_ids, all_seed_ids=all_seed_ids, max_iters=args.max_iters
            )

    # Save final log
    timestamp = time.strftime(r"%Y%m%d-%H%M%S")
    save_iteration_log(log_dir, adv_prompts, responses, scores, iters, timestamp, iteration=i, max_iters=args.max_iters)
    
    # Save final comprehensive log
    save_comprehensive_log(
        log_dir, all_prompts, all_responses, all_scores, all_similarities, rejection_reasons, timestamp, iteration=i,
        all_prompt_ids=all_prompt_ids, all_parent_ids=all_parent_ids, all_seed_ids=all_seed_ids, max_iters=args.max_iters
    )

    # Return final archives
    return adv_prompts, responses, scores


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration and seed prompts
    config = ConfigurationLoader.load(args.config_file)

    # Merge configuration with command-line arguments
    merged_args = merge_config_with_args(config, args)

    # Update configuration based on merged arguments
    # Only override dataset if not specified in config file
    if not config.sample_prompts:
        config.sample_prompts = merged_args.dataset

    # Initialize language models and scoring functions
    llms = initialize_language_models(config)
    fitness_fn = LlamaGuard(config.fitness_llm.model_kwargs)
    similarity_fn = BleuScoreNLTK()

    # Show configuration
    print(config)

    # Run the adversarial prompt generation process
    run_rainbowplus(
        merged_args,
        config,
        seed_prompts=[],
        llms=llms,
        fitness_fn=fitness_fn,
        similarity_fn=similarity_fn,
    )
