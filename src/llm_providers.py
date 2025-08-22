"""
LLM Provider abstraction for supporting multiple AI models.
Supports OpenAI and Google Gemini with easy switching.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate a response using the LLM."""
        pass
    
    @abstractmethod
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for the given texts."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider for GPT models and embeddings."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI provider initialized")
            except ImportError:
                logger.error("OpenAI package not installed. Run: pip install openai")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful board game recommendation expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the official Google Gen AI SDK."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self.embedding_model = None
        
        if self.api_key:
            try:
                from google import genai
                from google.genai import types
                
                # Initialize the client with API key
                self.client = genai.Client(api_key=self.api_key)
                self.types = types
                
                # For embeddings, we'll use sentence-transformers
                # The new SDK supports embeddings but we'll keep this for consistency
                try:
                    from sentence_transformers import SentenceTransformer
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("Gemini provider initialized with sentence-transformers for embeddings")
                except ImportError:
                    logger.warning("sentence-transformers not available, embeddings will be limited")
                
                logger.info("Gemini provider initialized with Google Gen AI SDK")
            except ImportError:
                logger.error("Google Gen AI SDK not installed. Run: pip install google-genai")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
    
    def generate_response(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        if not self.client:
            raise ValueError("Gemini client not initialized")
        
        try:
            # Use the new SDK's generate_content method
            response = self.client.models.generate_content(
                model='gemini-1.5-flash',  # or 'gemini-1.5-pro' for higher quality
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Option 1: Use sentence-transformers (current approach)
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.encode(texts)
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Embedding creation with sentence-transformers failed: {e}")
        
        # Option 2: Use Gemini's embedding model (if available)
        if self.client:
            try:
                # Note: The new SDK supports embeddings, but we'll need to check the exact API
                # For now, fallback to sentence-transformers
                logger.warning("Using sentence-transformers for embeddings (Gemini embeddings not implemented yet)")
                if self.embedding_model:
                    embeddings = self.embedding_model.encode(texts)
                    return embeddings.tolist()
            except Exception as e:
                logger.error(f"Gemini embedding creation failed: {e}")
        
        raise ValueError("No embedding method available")
    
    def is_available(self) -> bool:
        return self.client is not None and self.api_key is not None


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """
    Get an LLM provider based on configuration or available API keys.
    
    Args:
        provider_name: Specific provider to use ("openai" or "gemini")
                      If None, will auto-detect based on available API keys
    
    Returns:
        LLMProvider instance
    
    Raises:
        ValueError: If no providers are available
    """
    # If specific provider requested
    if provider_name:
        if provider_name.lower() == "openai":
            provider = OpenAIProvider()
            if provider.is_available():
                return provider
            else:
                raise ValueError("OpenAI provider requested but not available (check OPENAI_API_KEY)")
        
        elif provider_name.lower() in ["gemini", "google"]:
            provider = GeminiProvider()
            if provider.is_available():
                return provider
            else:
                raise ValueError("Gemini provider requested but not available (check GEMINI_API_KEY)")
        
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    # Auto-detect based on environment variable preference
    preferred_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if preferred_provider == "gemini":
        # Try Gemini first
        gemini = GeminiProvider()
        if gemini.is_available():
            logger.info("Using Gemini provider")
            return gemini
        
        # Fallback to OpenAI
        openai = OpenAIProvider()
        if openai.is_available():
            logger.info("Gemini not available, falling back to OpenAI")
            return openai
    
    else:
        # Try OpenAI first
        openai = OpenAIProvider()
        if openai.is_available():
            logger.info("Using OpenAI provider")
            return openai
        
        # Fallback to Gemini
        gemini = GeminiProvider()
        if gemini.is_available():
            logger.info("OpenAI not available, falling back to Gemini")
            return gemini
    
    # If we get here, no providers are available
    raise ValueError(
        "No LLM providers available. Please set either:\n"
        "- OPENAI_API_KEY for OpenAI\n"
        "- GEMINI_API_KEY for Gemini"
    )