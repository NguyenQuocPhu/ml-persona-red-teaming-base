
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class LLMConfig:
    """Configuration for a Language Model."""

    type_: str
    api_key: str = None
    base_url: str = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    sampling_params: Dict[str, Any] = field(default_factory=dict)
