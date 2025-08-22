"""
Demo script for the RAG-based board game recommendation system.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_engine import BoardGameRAG


async def main():
    """Demo the RAG system with example queries."""
    
    # Check for API keys
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not has_gemini and not has_openai:
        print("ERROR: No LLM API key found!")
        print("Please set one of these in your .env file:")
        print("  - GEMINI_API_KEY for Gemini (recommended)")
        print("  - OPENAI_API_KEY for OpenAI")
        print("\nGet a free Gemini API key from: https://aistudio.google.com/app/apikey")
        return
    
    print("🎲 Board Game RAG Demo")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        print(f"Initializing RAG system with {provider.upper()} provider...")
        rag = BoardGameRAG(provider_name=provider)
        
        # Initialize with dataset
        csv_path = os.path.join(os.path.dirname(__file__), "data", "bgg_dataset.csv")
        await asyncio.get_event_loop().run_in_executor(None, rag.initialize, csv_path)
        
        print("✅ RAG system initialized!\n")
        
        # Example queries
        example_queries = [
            "I want a strategic game for 2-4 players that takes about 90 minutes",
            "Show me some cooperative games that are good for beginners",
            "What are the best party games for 6+ players?",
            "I need a quick game that takes less than 30 minutes",
            "Recommend some highly-rated euro games with medium complexity"
        ]
        
        for i, query in enumerate(example_queries, 1):
            print(f"🔍 Query {i}: {query}")
            print("-" * 40)
            
            # Process query
            result = await asyncio.get_event_loop().run_in_executor(None, rag.query, query, 3)
            
            print(f"🤖 Response:")
            print(result['response'])
            print()
            
            print(f"📋 Top Games Found:")
            for j, game in enumerate(result['games'], 1):
                meta = game['metadata']
                print(f"  {j}. {meta['name']} ({meta['year']})")
                print(f"     Rating: {meta['rating']:.1f}/10, Complexity: {meta['complexity']:.1f}/5")
                print(f"     Players: {meta['min_players']}-{meta['max_players']}, Time: {meta['play_time']}min")
            
            print("\n" + "=" * 50 + "\n")
        
        # Interactive mode
        print("🎮 Interactive Mode - Enter your own queries!")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                user_query = input("Your query: ").strip()
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_query:
                    continue
                
                print("Processing...")
                result = await asyncio.get_event_loop().run_in_executor(None, rag.query, user_query, 5)
                
                print(f"\n🤖 Response:")
                print(result['response'])
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nThanks for trying the Board Game RAG system! 🎲")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())