"""
API calling utilities for OpenAI and Ollama models.
"""

import os
import time

import httpx

from lightrag.utils import logger


async def call_model(model_name: str, question: str, system_prompt: str, api_key: str = None, base_url: str = None) -> tuple:
    """
    Call a model (either OpenAI or Ollama) with the given question and system prompt.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4o-mini", "gemma2", etc.)
        question: The user question
        system_prompt: The system prompt with context
        api_key: OpenAI API key (only for OpenAI models)
        base_url: Base URL for API (only for OpenAI models)
    
    Returns:
        Tuple of (response_text, logprobs, input_tokens, output_tokens, inference_time_ms)
    """
    try:
        if model_name.startswith("gpt-"):
            # OpenAI model - use OpenAI SDK directly to ensure logprobs work
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})
            
            start_time = time.perf_counter()
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                logprobs=True,
                top_logprobs=1
            )
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            await client.close()
            
            # Extract content and logprobs
            content = response.choices[0].message.content
            lp = response.choices[0].logprobs
            
            # Extract token counts from usage
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            
            logprobs_data = None
            if lp and hasattr(lp, "content") and lp.content:
                logprobs_data = [
                    {"token": item.token, "logprob": item.logprob}
                    for item in lp.content
                ]
            
            return content, logprobs_data, input_tokens, output_tokens, inference_time_ms
        else:
            # Ollama model - use raw HTTP API (Python client doesn't support logprobs)
            ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            # Ensure URL has protocol prefix
            if ollama_url and not ollama_url.startswith(("http://", "https://")):
                ollama_url = f"http://{ollama_url}"
            
            start_time = time.perf_counter()
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": question,
                        "system": system_prompt,
                        "stream": False,
                        "options": {"num_predict": 2048},
                        "logprobs": True,
                        "top_logprobs": 1
                    }
                )
                resp.raise_for_status()
                result = resp.json()
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract logprobs from response
            logprobs_raw = result.get('logprobs')
            
            logprobs_data = None
            if logprobs_raw is not None:  # Distinguish None from empty list
                logprobs_data = [
                    {"token": lp.get("token", ""), "logprob": lp.get("logprob", 0.0)}
                    for lp in logprobs_raw
                ]
            
            # Ollama returns token counts in response
            input_tokens = result.get('prompt_eval_count', 0)
            output_tokens = result.get('eval_count', 0)
            
            response_text = result.get('response', '')
            return response_text, logprobs_data, input_tokens, output_tokens, inference_time_ms
    except Exception as e:
        logger.error(f"Error calling model {model_name}: {e}")
        return f"ERROR calling {model_name}: {str(e)}", None, 0, 0, 0
