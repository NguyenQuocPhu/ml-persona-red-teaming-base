
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
from google.colab import userdata

# Step 1: Load Gemini API key
try:
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
    os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
    print("✅ Gemini API key loaded and set in environment")
except Exception as e:
    print(f"❌ Failed to load Gemini API key: {e}")
    
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
