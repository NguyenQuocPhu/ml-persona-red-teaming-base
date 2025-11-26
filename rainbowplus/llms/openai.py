
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from openai import OpenAI
from rainbowplus.llms.base import BaseLLM
from pydantic import BaseModel
import time
import httpx
import openai
from typing import List


class LLMviaOpenAI(BaseLLM):
    def __init__(self, config):
        self.config = config
        self.model_kwargs = config.model_kwargs
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def get_name(self):
        return self.model_kwargs["model"]

    def generate(self, query: str, sampling_params: dict, max_retries: int = 5, backoff: float = 2.0):
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_kwargs["model"],
                    messages=[
                        {"role": "user", "content": query},
                    ],
                    **sampling_params
                )
                return response.choices[0].message.content
            except (openai.APIConnectionError, openai.RateLimitError, httpx.HTTPError, httpx.RemoteProtocolError) as e:
                last_exception = e
                print(f"OpenAI API error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = backoff * (2 ** attempt)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)  # Exponential backoff
                else:
                    print(f"All {max_retries} attempts failed. Raising final exception.")
                    raise e
        # If all retries fail, raise the last exception
        raise last_exception

    def batch_generate(self, queries: List[str], sampling_params: dict):
        responses = [self.generate(query, sampling_params) for query in queries]
        return responses

    def generate_format(
        self, query: str, sampling_params: dict, response_format: BaseModel
    ):
        completion = self.client.beta.chat.completions.parse(
            model=self.model_kwargs["model"],
            messages=[
                {"role": "user", "content": query},
            ],
            **sampling_params,
            response_format=response_format
        )
        message = completion.choices[0].message
        return message.parsed