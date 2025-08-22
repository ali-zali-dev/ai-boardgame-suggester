"""
Simplified LLM Provider abstraction using LlamaIndex integrations.
Replaces the complex custom provider system with LlamaIndex's built-in providers.
"""

import os
from typing import Optional
import logging

# LlamaIndex LLM imports (Gemini only)
# from llama_index.llms.openai import OpenAI  # COMMENTED OUT - OpenAI not needed
from llama_index.llms.gemini import Gemini

logger = logging.getLogger(__name__)


def get_llm_provider(provider_name: Optional[str] = None):
    """
    Get an LLM provider using LlamaIndex integrations (Gemini only).
    
    Args:
        provider_name: Specific provider to use ("gemini" only - OpenAI commented out)
                      If None, will use Gemini by default
    
    Returns:
        LlamaIndex Gemini LLM instance
    
    Raises:
        ValueError: If Gemini API key is not available
    """
    # Check available API keys (Gemini only)
    # has_openai = bool(os.getenv("OPENAI_API_KEY"))  # COMMENTED OUT - OpenAI not needed
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    
    if not has_gemini:
        raise ValueError(
            "No Gemini API key found! Please set:\n"
            "- GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini\n\n"
            "Get API key from:\n"
            "- Gemini: https://aistudio.google.com/app/apikey"
        )
    
    # If specific provider requested (Gemini only)
    if provider_name:
        # if provider_name.lower() == "openai":  # COMMENTED OUT - OpenAI not needed
        #     if not has_openai:
        #         raise ValueError("OpenAI provider requested but OPENAI_API_KEY not found")
        #     logger.info("Using OpenAI provider")
        #     return OpenAI(
        #         model="gpt-3.5-turbo",
        #         temperature=0.7,
        #         api_key=os.getenv("OPENAI_API_KEY")
        #     )
        
        if provider_name.lower() in ["gemini", "google"]:
            if not has_gemini:
                raise ValueError("Gemini provider requested but GEMINI_API_KEY not found")
            logger.info("Using Gemini provider")
            return Gemini(
                model="models/gemini-1.5-flash",
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider_name}. Only 'gemini' is supported (OpenAI commented out)")
    
    # Auto-detect based on environment variable preference (Gemini only)
    preferred_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if preferred_provider == "gemini" and has_gemini:
        logger.info("Using Gemini provider")
        return Gemini(
            model="models/gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
    # elif preferred_provider == "openai" and has_openai:  # COMMENTED OUT - OpenAI not needed
    #     logger.info("Using OpenAI provider")
    #     return OpenAI(
    #         model="gpt-3.5-turbo",
    #         temperature=0.7,
    #         api_key=os.getenv("OPENAI_API_KEY")
    #     )
    elif has_gemini:
        logger.info("Using Gemini provider (fallback)")
        return Gemini(
            model="models/gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
    # elif has_openai:  # COMMENTED OUT - OpenAI not needed
    #     logger.info("Using OpenAI provider (fallback)")
    #     return OpenAI(
    #         model="gpt-3.5-turbo",
    #         temperature=0.7,
    #         api_key=os.getenv("OPENAI_API_KEY")
    #     )
    
    # If we get here, Gemini is not available
    raise ValueError(
        "No Gemini provider available. Please set:\n"
        "- GEMINI_API_KEY for Gemini"
    )


# Backwards compatibility aliases (OpenAI commented out)
class LLMProvider:
    """Backwards compatibility class."""
    pass

# class OpenAIProvider(LLMProvider):  # COMMENTED OUT - OpenAI not needed
#     """Backwards compatibility - use get_llm_provider('openai') instead."""
#     def __init__(self, *args, **kwargs):
#         raise DeprecationWarning("Use get_llm_provider('openai') instead")

class GeminiProvider(LLMProvider):
    """Backwards compatibility - use get_llm_provider('gemini') instead."""
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning("Use get_llm_provider('gemini') instead")


if __name__ == "__main__":
    # Test the provider system
    try:
        llm = get_llm_provider()
        print(f"✅ LLM Provider initialized: {type(llm).__name__}")
        print(f"Model: {llm.model}")
    except Exception as e:
        print(f"❌ Error: {e}")