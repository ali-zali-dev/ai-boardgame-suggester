"""
Main runner for the RAG-based board game recommendation API.
"""

import uvicorn
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_api import app

if __name__ == "__main__":
    # Check for API keys (Gemini only)
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    # has_openai = bool(os.getenv("OPENAI_API_KEY"))  # COMMENTED OUT - OpenAI not needed
    
    if not has_gemini:
        print("ERROR: No Gemini API key found!")
        print("Please set this in your .env file:")
        print("  - GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini")
        print("\nGet a free Gemini API key from: https://aistudio.google.com/app/apikey")
        print("Example .env file:")
        print("  GEMINI_API_KEY=your-gemini-api-key-here")
        print("  LLM_PROVIDER=gemini")
        sys.exit(1)
    
    # Show which provider will be used
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    print(f"Starting Board Game RAG API with {provider.upper()} provider...")
    print("API Documentation available at: http://localhost:8000/docs")
    print("Health check available at: http://localhost:8000/health")
    
    if has_gemini and provider == "gemini":
        print("✅ Using Gemini (Google AI)")
    # elif has_openai and provider == "openai":  # COMMENTED OUT - OpenAI not needed
    #     print("✅ Using OpenAI")
    elif has_gemini:
        print("✅ Using Gemini (Google AI) as fallback")
    # elif has_openai:  # COMMENTED OUT - OpenAI not needed
    #     print("✅ Using OpenAI as fallback")
    
    uvicorn.run(
        "src.rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )