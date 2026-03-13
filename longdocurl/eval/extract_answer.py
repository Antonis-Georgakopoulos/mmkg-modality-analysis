"""
Answer extraction for LongDocURL benchmark.
Uses an LLM to extract concise answers from detailed responses.
"""

from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


def extract_answer(question, output, prompt, model_name="gpt-5.1", api_key=None, base_url=None):
    """
    Extract a concise answer from a detailed model response using an LLM.
    
    Args:
        question: The original question
        output: The detailed response from the model
        prompt: The extraction prompt template
        model_name: Model to use for extraction
        api_key: API key for the model
        base_url: Base URL for the API
        
    Returns:
        Extracted answer string
    """
    try:
        # Create client with API key and base_url if provided
        if api_key:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI()
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                "role": "assistant",
                "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)
                }
            ],
            temperature=0.0,
            max_tokens=256
        )
        response = response.choices[0].message.content
        logger.debug(f"Extract answer response: {response[:100]}...")
    except Exception as e:
        logger.error(f"Error in extract_answer: {e}")
        response = "Failed"
    
    return response
