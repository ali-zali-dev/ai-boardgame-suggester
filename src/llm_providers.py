"""
Simplified LLM Provider abstraction using LlamaIndex integrations.
Replaces the complex custom provider system with LlamaIndex's built-in providers.
"""

import os
from typing import Optional
import logging

# LlamaIndex LLM imports
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

logger = logging.getLogger(__name__)


def get_llm_provider(provider_name: Optional[str] = None):
    """
    Get an LLM provider using LlamaIndex integrations.
    
    Args:
        provider_name: Specific provider to use ("openai" or "anthropic")
                      If None, will auto-detect based on available API keys
    
    Returns:
        LlamaIndex LLM instance
    
    Raises:
        ValueError: If no providers are available
    """
    # Check available API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    
    if not has_openai and not has_gemini:
        raise ValueError(
            "No LLM API keys found! Please set either:\n"
            "- OPENAI_API_KEY for OpenAI\n"
            "- GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini\n\n"
            "Get API keys from:\n"
            "- OpenAI: https://platform.openai.com/api-keys\n"
            "- Gemini: https://aistudio.google.com/app/apikey"
        )
    
    # If specific provider requested
    if provider_name:
        if provider_name.lower() == "openai":
            if not has_openai:
                raise ValueError("OpenAI provider requested but OPENAI_API_KEY not found")
            logger.info("Using OpenAI provider")
            return OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        elif provider_name.lower() in ["gemini", "google"]:
            if not has_gemini:
                raise ValueError("Gemini provider requested but GEMINI_API_KEY not found")
            logger.info("Using Gemini provider")
            return Gemini(
                model="models/gemini-1.5-flash",
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider_name}. Use 'openai' or 'gemini'")
    
    # Auto-detect based on environment variable preference
    preferred_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if preferred_provider == "gemini" and has_gemini:
        logger.info("Using Gemini provider")
        return Gemini(
            model="models/gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
    elif preferred_provider == "openai" and has_openai:
        logger.info("Using OpenAI provider")
        return OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif has_gemini:
        logger.info("Using Gemini provider (fallback)")
        return Gemini(
            model="models/gemini-1.5-flash",
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
    elif has_openai:
        logger.info("Using OpenAI provider (fallback)")
        return OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # If we get here, no providers are available
    raise ValueError(
        "No LLM providers available. Please set either:\n"
        "- OPENAI_API_KEY for OpenAI\n"
        "- GEMINI_API_KEY for Gemini"
    )


# Backwards compatibility aliases
class LLMProvider:
    """Backwards compatibility class."""
    pass

class OpenAIProvider(LLMProvider):
    """Backwards compatibility - use get_llm_provider('openai') instead."""
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning("Use get_llm_provider('openai') instead")

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