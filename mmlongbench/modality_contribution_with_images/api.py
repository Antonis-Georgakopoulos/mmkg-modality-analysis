"""
API calling utilities for OpenAI and Ollama models (WITH IMAGE SUPPORT).
"""

import os
import time
import base64
from pathlib import Path
from typing import List, Dict, Optional

import httpx

from lightrag.utils import logger


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string or None if file doesn't exist/can't be read
    """
    try:
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None
        
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type based on file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    return mime_types.get(ext, 'image/jpeg')


async def call_model(model_name: str, question: str, system_prompt: str, api_key: str = None, base_url: str = None) -> tuple:
    """
    Call a model (either OpenAI or Ollama) with the given question and system prompt.
    Text-only version for backward compatibility.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4o-mini", "gemma2", etc.)
        question: The user question
        system_prompt: The system prompt with context
        api_key: OpenAI API key (only for OpenAI models)
        base_url: Base URL for API (only for OpenAI models)
    
    Returns:
        Tuple of (response_text, logprobs, input_tokens, output_tokens, inference_time_ms)
    """
    # Delegate to the image-capable version with no images
    return await call_model_with_images(
        model_name=model_name,
        question=question,
        system_prompt=system_prompt,
        images=[],
        api_key=api_key,
        base_url=base_url
    )


async def call_model_with_images(
    model_name: str,
    question: str,
    system_prompt: str,
    images: List[Dict[str, str]],
    api_key: str = None,
    base_url: str = None
) -> tuple:
    """
    Call a model with the given question, system prompt, and optional images.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4o", "llava", etc.)
        question: The user question
        system_prompt: The system prompt with context
        images: List of dicts with {"base64": str, "mime_type": str} for each image
        api_key: OpenAI API key (only for OpenAI models)
        base_url: Base URL for API (only for OpenAI models)
    
    Returns:
        Tuple of (response_text, logprobs, input_tokens, output_tokens, inference_time_ms)
    """
    try:
        if model_name.startswith("gpt-"):
            return await _call_openai_with_images(
                model_name, question, system_prompt, images, api_key, base_url
            )
        else:
            return await _call_ollama_with_images(
                model_name, question, system_prompt, images
            )
    except Exception as e:
        logger.error(f"Error calling model {model_name}: {e}")
        return f"ERROR calling {model_name}: {str(e)}", None, 0, 0, 0


async def _call_openai_with_images(
    model_name: str,
    question: str,
    system_prompt: str,
    images: List[Dict[str, str]],
    api_key: str,
    base_url: str
) -> tuple:
    """Call OpenAI API with images (vision model)."""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Build user message with images
    if images:
        # Multimodal content: text + images
        content = []
        content.append({"type": "text", "text": question})
        
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['mime_type']};base64,{img['base64']}",
                    "detail": "high"  # Use high detail for better accuracy
                }
            })
        
        messages.append({"role": "user", "content": content})
        logger.info(f"📸 Sending {len(images)} images to {model_name}")
    else:
        # Text-only
        messages.append({"role": "user", "content": question})
    
    start_time = time.perf_counter()
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        logprobs=True,
        top_logprobs=1,
        max_tokens=4096
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


async def call_model_with_vlm_messages(
    model_name: str,
    messages: List[Dict],
    api_key: str = None,
    base_url: str = None
) -> tuple:
    """
    Call a model with pre-built VLM messages (matching raganything approach).
    
    Args:
        model_name: Name of the model
        messages: Pre-built messages list with interleaved text and images
        api_key: OpenAI API key
        base_url: Base URL for API
    
    Returns:
        Tuple of (response_text, logprobs, input_tokens, output_tokens, inference_time_ms)
    """
    # ===== TEMPORARY LOG: Verify images in VLM messages =====
    logger.info("=" * 60)
    logger.info("🔍 API CALL: call_model_with_vlm_messages")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Number of messages: {len(messages)}")
    
    # Count images in messages
    image_count = 0
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        image_count += 1
                        # Log image size (base64 length as proxy)
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            base64_len = len(url.split(",", 1)[1]) if "," in url else 0
                            logger.info(f"   📷 Image found: base64 length = {base64_len} chars (~{base64_len * 3 // 4 // 1024} KB)")
    
    logger.info(f"   Total images in messages: {image_count}")
    if image_count == 0:
        logger.warning("   ⚠️ NO IMAGES IN API CALL - text-only mode")
    else:
        logger.info(f"   ✅ IMAGES WILL BE SENT TO MODEL: {image_count} image(s)")
    logger.info("=" * 60)
    
    try:
        if model_name.startswith("gpt-"):
            return await _call_openai_with_vlm_messages(model_name, messages, api_key, base_url)
        else:
            return await _call_ollama_with_vlm_messages(model_name, messages)
    except Exception as e:
        logger.error(f"Error calling model {model_name} with VLM messages: {e}")
        return f"ERROR calling {model_name}: {str(e)}", None, 0, 0, 0


async def _call_openai_with_vlm_messages(
    model_name: str,
    messages: List[Dict],
    api_key: str,
    base_url: str
) -> tuple:
    """Call OpenAI with pre-built VLM messages."""
    from openai import AsyncOpenAI
    
    logger.info(f"🚀 Sending request to OpenAI {model_name}...")
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    start_time = time.perf_counter()
    response = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        logprobs=True,
        top_logprobs=1,
        max_tokens=4096
    )
    inference_time_ms = (time.perf_counter() - start_time) * 1000
    
    await client.close()
    
    # Extract content and logprobs
    content = response.choices[0].message.content
    lp = response.choices[0].logprobs
    
    # Extract token counts from usage
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    
    # ===== TEMPORARY LOG: Verify API response =====
    logger.info("✅ OpenAI response received:")
    logger.info(f"   Input tokens: {input_tokens}")
    logger.info(f"   Output tokens: {output_tokens}")
    logger.info(f"   Inference time: {inference_time_ms:.0f}ms")
    logger.info(f"   Response preview: {content[:200] if content else 'EMPTY'}...")
    
    logprobs_data = None
    if lp and hasattr(lp, "content") and lp.content:
        logprobs_data = [
            {"token": item.token, "logprob": item.logprob}
            for item in lp.content
        ]
    
    return content, logprobs_data, input_tokens, output_tokens, inference_time_ms


async def _call_ollama_with_vlm_messages(
    model_name: str,
    messages: List[Dict]
) -> tuple:
    """Call Ollama with pre-built VLM messages."""
    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if ollama_url and not ollama_url.startswith(("http://", "https://")):
        ollama_url = f"http://{ollama_url}"
    
    # Convert messages to Ollama format
    ollama_messages = []
    for msg in messages:
        if msg["role"] == "system":
            ollama_messages.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, list):
                # Multimodal content - extract text and images
                text_parts = []
                images = []
                for part in content:
                    if part["type"] == "text":
                        text_parts.append(part["text"])
                    elif part["type"] == "image_url":
                        # Extract base64 from data URL
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            base64_data = url.split(",", 1)[1]
                            images.append(base64_data)
                
                ollama_messages.append({
                    "role": "user",
                    "content": "\n".join(text_parts),
                    "images": images
                })
            else:
                ollama_messages.append({"role": "user", "content": content})
    
    start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model_name,
                "messages": ollama_messages,
                "stream": False,
                "options": {"num_predict": 2048, "num_ctx": 32768},
                "logprobs": True,
                "top_logprobs": 1
            }
        )
        resp.raise_for_status()
        result = resp.json()
    
    inference_time_ms = (time.perf_counter() - start_time) * 1000
    
    response_text = result.get('message', {}).get('content', '')
    input_tokens = result.get('prompt_eval_count', 0)
    output_tokens = result.get('eval_count', 0)
    
    # Extract logprobs from top-level result
    logprobs_data = None
    logprobs_raw = result.get('logprobs')
    if logprobs_raw:
        logprobs_data = [
            {"token": lp.get("token", ""), "logprob": lp.get("logprob", 0.0)}
            for lp in logprobs_raw
        ]
    
    return response_text, logprobs_data, input_tokens, output_tokens, inference_time_ms


async def _call_ollama_with_images(
    model_name: str,
    question: str,
    system_prompt: str,
    images: List[Dict[str, str]]
) -> tuple:
    """Call Ollama API with images (vision model like llava)."""
    ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # Ensure URL has protocol prefix
    if ollama_url and not ollama_url.startswith(("http://", "https://")):
        ollama_url = f"http://{ollama_url}"
    
    start_time = time.perf_counter()
    
    if images:
        # Use /api/chat endpoint for multimodal (images)
        # Ollama expects images as base64 strings in the "images" field
        image_base64_list = [img['base64'] for img in images]
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({
            "role": "user",
            "content": question,
            "images": image_base64_list
        })
        
        logger.info(f"📸 Sending {len(images)} images to Ollama {model_name}")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": 2048, "num_ctx": 32768},
                    "logprobs": True,
                    "top_logprobs": 1
                }
            )
            resp.raise_for_status()
            result = resp.json()
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Extract response from chat format
        response_text = result.get('message', {}).get('content', '')
        
        # Extract logprobs from top-level result
        logprobs_data = None
        logprobs_raw = result.get('logprobs')
        if logprobs_raw:
            logprobs_data = [
                {"token": lp.get("token", ""), "logprob": lp.get("logprob", 0.0)}
                for lp in logprobs_raw
            ]
        
        # Token counts
        input_tokens = result.get('prompt_eval_count', 0)
        output_tokens = result.get('eval_count', 0)
        
    else:
        # Text-only: use /api/generate endpoint
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": question,
                    "system": system_prompt,
                    "stream": False,
                    "options": {"num_predict": 2048, "num_ctx": 32768},
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
        if logprobs_raw is not None:
            logprobs_data = [
                {"token": lp.get("token", ""), "logprob": lp.get("logprob", 0.0)}
                for lp in logprobs_raw
            ]
        
        # Token counts
        input_tokens = result.get('prompt_eval_count', 0)
        output_tokens = result.get('eval_count', 0)
        
        response_text = result.get('response', '')
    
    return response_text, logprobs_data, input_tokens, output_tokens, inference_time_ms
