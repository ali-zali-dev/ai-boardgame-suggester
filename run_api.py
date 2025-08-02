#!/usr/bin/env python3
"""
Startup script for the Board Game Recommendation API.
Runs the FastAPI server with uvicorn.
"""

import uvicorn
import sys
import os

# Add src to Python path
sys.path.insert(0, 'src')

def main():
    """Run the FastAPI application."""
    print("🎲 Starting Board Game Recommendation API...")
    print("📊 Initializing recommendation system (this may take a moment)...")
    print("🚀 API will be available at: http://localhost:8000")
    print("📚 Swagger documentation: http://localhost:8000/docs")
    print("📖 ReDoc documentation: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "src.api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["src"],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()