
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import yaml

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
from yaml.parser import ParserError

from rainbowplus.configs.base import LLMConfig


@dataclass
class Configuration:
    """Main configuration class containing all settings."""

    target_llm: LLMConfig
    fitness_llm: LLMConfig
    archive: Dict[str, Any] = field(default_factory=dict)
    sample_prompts: str = ""
    seed_memory: str = ""
    mutator_llm: Optional[LLMConfig] = None
    mutation_strategy: Optional[str] = None
    persona_config: Optional[str] = None
    # Added, default to experts, PersonaMutator ("RedTeamingExperts" | "RegularAIUsers")
    persona_type: Optional[str] = "RedTeamingExperts" 
    log_dir: str = "./logs"
    num_samples: int = 150
    n_cells: int = 100
    max_iters: int = 1000
    sim_threshold: float = 0.6
    num_mutations: int = 5
    fitness_threshold: float = 0.5
    log_interval: int = 50
    shuffle: bool = True
    threshold_bot: float = 0.4
    threshold_top: float = 0.6
    number_example_prompts: int = 2

    def __post_init__(self):
        self.mutator_llm = self.mutator_llm or self.target_llm


class ConfigurationLoader:
    @staticmethod
    def load(file_path: str) -> Configuration:
        """Load configuration from YAML file."""
        try:
            with open(file_path, encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
                print(data)
                return ConfigurationLoader._create_config(data)
        except ParserError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except OSError as e:
            raise ValueError(f"Failed to read config file: {e}")

    @staticmethod
    def _create_config(data: Dict[str, Any]) -> Configuration:
        """Create Configuration instance from parsed YAML data."""
        required_llms = ["target_llm", "fitness_llm"]

        try:
            llm_configs = {
                key: ConfigurationLoader._parse_llm_config(data.get(key, {}))
                for key in required_llms
            }

            return Configuration(
                archive=data.get("archive", {}),
                sample_prompts=data.get("sample_prompts", ""),
                mutator_llm=ConfigurationLoader._parse_llm_config(
                    data.get("mutator_llm", {})
                ),
                mutation_strategy=data.get("mutation_strategy"),
                persona_config=data.get("persona_config"),
                persona_type=data.get("persona_type", "RedTeamingExperts"),
                log_dir=data.get("log_dir", "./logs"),
                num_samples=data.get("num_samples", 150),
                max_iters=data.get("max_iters", 1000),
                sim_threshold=data.get("sim_threshold", 0.6),
                num_mutations=data.get("num_mutations", 5),
                fitness_threshold=data.get("fitness_threshold", 0.5),
                log_interval=data.get("log_interval", 50),
                shuffle=data.get("shuffle", True),
                **llm_configs,
            )
        except KeyError as e:
            raise ValueError(f"Missing required configuration: {e}")

    @staticmethod
    def _parse_llm_config(data: Any) -> LLMConfig:
        """Parse LLM configuration from YAML data."""
        if not isinstance(data, dict):
            return None
        return LLMConfig(**data)
