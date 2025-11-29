# gemini_llm.py
# Rewritten GeminiLLM class (2025-compatible)
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import os
import time
import random
import logging
from typing import Any, Dict, List

import google.generativeai as genai
from google.api_core.exceptions import (
    GoogleAPIError,
    ResourceExhausted,
    InternalServerError,
    ServiceUnavailable,
    InvalidArgument,
)

# Optional Colab userdata import (safe fallback)
try:
    from google.colab import userdata  # type: ignore
except Exception:
    userdata = None  # type: ignore

# Import your project's BaseLLM; adjust import path if needed
try:
    from rainbowplus.llms.base import BaseLLM
except Exception:
    # Minimal fallback if BaseLLM isn't available — keeps this file runnable for testing
    class BaseLLM:
        def __init__(self):
            pass


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GeminiLLM(BaseLLM):
    """
    Gemini wrapper with robust environment handling, rate-limit-aware retries,
    and safe exception handling for the 2024-2025 Google Gemini SDK.

    Usage notes:
    - Provide API key via environment variable GEMINI_API_KEY or Colab userdata.
    - If no key is found, the class will initialise but generate() will return "" and log a warning.
    - This class does NOT embed or leak any API keys.
    """

    DEFAULT_RATE_LIMITS = {"rpm": 10, "tpm": 100000, "rpd": 100}

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        # Acquire API key from environment or Colab userdata (if present)
        #self.api_key = self._get_api_key()
        genai.configure(api_key = self.config.api_key)
        """
        if self.api_key:
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured from environment/colab.")
        else:
            logger.warning(
                "No Gemini API key found. GeminiLLM will not be able to call the remote API."
            )
        """
        # Model name and client wrapper
        self.model_name = (
            getattr(config, "model_kwargs", {}).get("model")
            if getattr(config, "model_kwargs", None)
            else None
        ) or "gemini-1.5-flash"

        # Defer creation of model wrapper until first use (avoid raising on init)
        self._model = None

        # Rate limit bookkeeping
        self.request_timestamps: List[float] = []
        self.total_tokens_used = 0
        self.rate_limit_reset_time = time.time() + 60

        # Configure model-specific rate limits
        self.rate_limits = self._determine_rate_limits(self.model_name)
        logger.info(f"Configured rate limits for {self.model_name}: {self.rate_limits}")

    def _get_api_key(self) -> str:
        """Try multiple places for an API key without hardcoding secrets."""
        # 1) Explicit environment variable
        key = os.getenv("GEMINI_API_KEY")
        if key:
            return key

        # 2) Colab userdata (if running in Colab and key was set there)
        if userdata is not None:
            try:
                k = userdata.get("GEMINI_API_KEY")
                if k:
                    return k
            except Exception:
                # Don't fail if Colab internals are not present
                pass

        # 3) No key
        return ""

    def _determine_rate_limits(self, model_name: str) -> Dict[str, int]:
        """Return conservative, reasonable rate limits for common Gemini models."""
        model_limits = {
            "gemini-2.5-flash": {"rpm": 10, "tpm": 250000, "rpd": 250},
            "gemini-2.5-pro": {"rpm": 2, "tpm": 125000, "rpd": 50},
            "gemini-1.5-flash": {"rpm": 15, "tpm": 1000000, "rpd": 200},
        }
        return model_limits.get(model_name, self.DEFAULT_RATE_LIMITS.copy())

    def _estimate_tokens(self, text: str) -> int:
        """Simple, conservative token estimation. Override if you have a tokenizer.
        Approximation: 4 characters ≈ 1 token.
        """
        return max(1, len(text) // 4)

    def _calculate_backoff(self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
        logger.debug(f"Backoff attempt {attempt}: waiting {delay:.2f}s")
        return delay

    def _check_rate_limits_and_maybe_sleep(self) -> None:
        """Check recent request history vs rate_limits and sleep if needed (non-strict).
        This is a best-effort client-side throttle to reduce likelihood of 429s.
        """
        now = time.time()
        # Keep timestamps for a rolling 60s window
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]

        # If we're near rpm, sleep until oldest timestamp falls out of window
        rpm = self.rate_limits.get("rpm", self.DEFAULT_RATE_LIMITS["rpm"])
        threshold = int(rpm * 0.8)
        if len(self.request_timestamps) >= threshold and len(self.request_timestamps) > 0:
            time_until_next = 60 - (now - min(self.request_timestamps))
            if time_until_next > 0:
                logger.warning(f"Approaching RPM client-side limit; sleeping {time_until_next:.1f}s")
                time.sleep(time_until_next)

        # Reset tokens per minute when window expires
        if now > self.rate_limit_reset_time:
            self.total_tokens_used = 0
            self.rate_limit_reset_time = now + 60

    def _ensure_model_client(self):
        """Initialize the genai model wrapper lazily (avoids raising on import/init).
        This uses genai.GenerativeModel(model_name) when an API key is present.
        """
        if self._model is None and self.api_key:
            try:
                self._model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                logger.exception("Failed to create Gemini GenerativeModel client: %s", e)
                self._model = None

    def get_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, sampling_params: Dict[str, Any] = None, max_retries: int = 4) -> str:
        """Generate text from Gemini with robust error handling and retries.

        Returns an empty string on failure (does not raise) to allow caller resilience.
        """
        if not self.api_key:
            logger.error("No API key configured for GeminiLLM; skipping remote call.")
            return ""

        self._ensure_model_client()
        if not self._model:
            logger.error("Gemini model client not available; cannot generate.")
            return ""

        # Merge sampling params with config defaults (if available)
        default_params = {}
        try:
            default_params = dict(getattr(self.config, "sampling_params", {}) or {})
        except Exception:
            default_params = {}

        if sampling_params:
            default_params.update(sampling_params)

        generation_config = genai.GenerationConfig(
            temperature=default_params.get("temperature", 0.7),
            top_p=default_params.get("top_p", 0.9),
            max_output_tokens=default_params.get("max_tokens", 512),
            top_k=default_params.get("top_k", None),
        )

        # Pre-check and client-side throttling
        self._check_rate_limits_and_maybe_sleep()

        last_exc = None
        for attempt in range(max_retries):
            try:
                # Bookkeeping
                now = time.time()
                self.request_timestamps.append(now)
                estimated_input_tokens = self._estimate_tokens(prompt)
                self.total_tokens_used += estimated_input_tokens

                # Token threshold check
                tpm = self.rate_limits.get("tpm", self.DEFAULT_RATE_LIMITS["tpm"])
                if self.total_tokens_used >= int(tpm * 0.9):
                    logger.warning("Client-side approaching declared TPM; sleeping until reset.")
                    sleep_for = max(0, self.rate_limit_reset_time - time.time())
                    if sleep_for > 0:
                        time.sleep(sleep_for)

                # Call Gemini
                resp = self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    # safety_settings=self.safety_settings,  # Uncomment if you want explicit safety overrides
                )

                # Extract text from response robustly
                text = ""
                if getattr(resp, "text", None):
                    text = resp.text
                elif getattr(resp, "candidates", None):
                    # Some SDK responses provide candidates list
                    try:
                        cand = resp.candidates[0]
                        text = getattr(cand, "output", None) or getattr(cand, "content", None) or getattr(cand, "text", "")
                    except Exception:
                        text = ""

                # Count output tokens conservatively
                if text:
                    self.total_tokens_used += self._estimate_tokens(text)
                    return text

                # If empty, raise to trigger retry/backoff
                raise ValueError("Empty response from Gemini API")

            except (ResourceExhausted, InternalServerError, ServiceUnavailable, GoogleAPIError, InvalidArgument) as e:
                last_exc = e
                logger.warning("Gemini API error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    backoff = self._calculate_backoff(attempt, base_delay=1.0)
                    logger.info("Retrying after %.2fs", backoff)
                    time.sleep(backoff)
                    continue
                else:
                    logger.error("Max retries reached. Last exception: %s", e)
                    break

            except Exception as e:
                # Catch content-safety exceptions from SDK if present under genai.types
                try:
                    if hasattr(genai, "types"):
                        t = genai.types
                        if getattr(t, "StopCandidateException", None) and isinstance(e, t.StopCandidateException):
                            logger.warning("Content blocked (StopCandidateException): %s", e)
                            return ""
                        if getattr(t, "BlockedPromptException", None) and isinstance(e, t.BlockedPromptException):
                            logger.warning("Content blocked (BlockedPromptException): %s", e)
                            return ""
                except Exception:
                    pass

                last_exc = e
                logger.exception("Unexpected error while calling Gemini: %s", e)
                break

        # Final failure path — return empty string but log for debugging
        logger.debug("Generate failed. Last exception: %s", last_exc)
        return ""

    def batch_generate(self, prompts: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        results: List[str] = []
        total = len(prompts)
        for idx, p in enumerate(prompts):
            logger.info("Generating %d/%d", idx + 1, total)
            # stagger requests to reduce 429s
            if idx > 0:
                # small dynamic delay based on rpm
                rpm = self.rate_limits.get("rpm", self.DEFAULT_RATE_LIMITS["rpm"])
                delay = max(0.2, 60.0 / max(1, rpm) / 2.0)
                time.sleep(delay)

            res = self.generate(p, sampling_params)
            results.append(res)
        return results

    @property
    def safety_settings(self) -> Dict[str, Any]:
        """Default safety settings; left conservative. If you are red-teaming you can override.
        This example returns an empty dict (SDK default behavior). Customize carefully.
        """
        return {}

    def get_rate_limit_status(self) -> Dict[str, Any]:
        now = time.time()
        requests_last_minute = len([ts for ts in self.request_timestamps if now - ts < 60])
        rpm = self.rate_limits.get("rpm", self.DEFAULT_RATE_LIMITS["rpm"])
        tpm = self.rate_limits.get("tpm", self.DEFAULT_RATE_LIMITS["tpm"])
        return {
            "model": self.model_name,
            "requests_last_minute": requests_last_minute,
            "max_requests_per_minute": rpm,
            "tokens_used_this_minute": self.total_tokens_used,
            "max_tokens_per_minute": tpm,
            "time_until_reset": max(0, self.rate_limit_reset_time - now),
            "rate_limit_utilization": {
                "rpm": (requests_last_minute / max(1, rpm)) * 100,
                "tpm": (self.total_tokens_used / max(1, tpm)) * 100,
            },
        }
