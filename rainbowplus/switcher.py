#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from typing import Any, Dict, List
from rainbowplus.llms.vllm import vLLM
from rainbowplus.llms.openai import LLMviaOpenAI
from rainbowplus.llms.gemini import GeminiLLM
import os
import logging

logger = logging.getLogger(__name__)

class LLMSwitcher:
    """Factory class for creating and managing different LLM implementations"""

    def __init__(self, config: Any):
        """
        Initialize LLM switcher with configuration

        Args:
            config: Configuration for the Language Model
        """
        self._type = config.type_
        self.config = config
        self.model_kwargs = config.model_kwargs
        self._llm = self._create_llm()

    def _get_gemini_api_key(self):
        """Safely get Gemini API key from multiple sources"""
        # Method 1: Check if API key is already in config
        if hasattr(self.config, 'api_key') and self.config.api_key:
            return self.config.api_key
        
        # Method 2: Try environment variable
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            return api_key
        
        # Method 3: Try Colab userdata (with proper error handling)
        try:
            from google.colab import userdata
            api_key = userdata.get('GEMINI_API_KEY')
            if api_key:
                logger.info("✅ Loaded Gemini API key from Colab secrets")
                return api_key
        except Exception as e:
            logger.warning(f"⚠️ Could not access Colab secrets: {e}")
        
        return None

    def _create_llm(self):
        """
        Create the appropriate LLM based on configuration

        Returns:
            Instantiated Language Model

        Raises:
            ValueError: If an unsupported LLM type is specified
        """
        if self._type == "vllm":
            return vLLM(self.model_kwargs)
        elif self._type == "openai":
            return LLMviaOpenAI(self.config)
        elif self._type == "gemini":
            # Get API key safely and add it to config
            api_key = self._get_gemini_api_key()
            if api_key:
                self.config.api_key = api_key
            else:
                logger.warning("⚠️ No Gemini API key found. LLM may fail to initialize.")
            return GeminiLLM(self.config)
        else:
            raise ValueError(f"Unsupported LLM type: {self._type}")

    def generate(self, query: str, sampling_params: Dict[str, Any] = None) -> str:
        """
        Generate a response using the selected LLM

        Args:
            query: Input query to generate response for
            sampling_params: Optional sampling parameters

        Returns:
            Generated response
        """
        if sampling_params is None:
            sampling_params = {}
        return self._llm.generate(query, sampling_params)

    def batch_generate(
        self, queries: List[str], sampling_params: Dict[str, Any] = None
    ) -> List[str]:
        """
        Generate responses for a batch of queries using the selected LLM

        Args:
            queries: List of input queries
            sampling_params: Optional sampling parameters

        Returns:
            List of generated responses
        """
        if sampling_params is None:
            sampling_params = {}
        return self._llm.batch_generate(queries, sampling_params)

    def generate_format(
        self,
        query: str,
        sampling_params: Dict[str, Any] = None,
        response_format: Any = None,
    ):
        return self._llm.generate_format(query, sampling_params, response_format)

    def __repr__(self) -> str:
        return f"LLMSwitcher(type={self._type}, model_kwargs={self.model_kwargs})"