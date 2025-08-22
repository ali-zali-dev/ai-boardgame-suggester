"""
Comprehensive test suite for the AI Board Game Suggester.
This file combines all testing functionality:
- LLM Provider testing
- Game matching and search testing  
- Description generation examples

Run this to verify your entire system is working correctly.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.llm_providers import get_llm_provider, OpenAIProvider, GeminiProvider
from src.rag_engine import BoardGameRAG


# =============================================================================
# PROVIDER TESTS (from test_providers.py)
# =============================================================================

def test_provider(provider_name: str):
    """Test a specific provider."""
    print(f"\n🧪 Testing {provider_name.upper()} Provider")
    print("-" * 40)
    
    try:
        if provider_name.lower() == "gemini":
            provider = GeminiProvider()
        elif provider_name.lower() == "openai":
            provider = OpenAIProvider()
        else:
            print(f"❌ Unknown provider: {provider_name}")
            return False
        
        # Check if available
        if not provider.is_available():
            print(f"❌ {provider_name} provider not available (check API key)")
            return False
        
        print(f"✅ {provider_name} provider initialized successfully")
        
        # Test embeddings
        print("Testing embeddings...")
        test_texts = ["This is a strategy game", "This is a party game"]
        embeddings = provider.create_embeddings(test_texts)
        print(f"✅ Created embeddings: {len(embeddings)} texts → {len(embeddings[0])} dimensions")
        
        # Test text generation
        print("Testing text generation...")
        response = provider.generate_response(
            "Recommend a board game for 2 players in one sentence.",
            max_tokens=50,
            temperature=0.7
        )
        print(f"✅ Generated response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing {provider_name}: {e}")
        return False


def run_provider_tests():
    """Test all available providers."""
    print("🎯 LLM Provider Test")
    print("=" * 50)
    
    # Check which API keys are available
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    print(f"Available API keys:")
    print(f"  Gemini: {'✅' if has_gemini else '❌'}")
    print(f"  OpenAI: {'✅' if has_openai else '❌'}")
    
    if not has_gemini and not has_openai:
        print("\n❌ No API keys found!")
        print("Please set either:")
        print("  - GEMINI_API_KEY for Gemini")
        print("  - OPENAI_API_KEY for OpenAI")
        return False
    
    success_count = 0
    total_tests = 0
    
    # Test available providers
    if has_gemini:
        total_tests += 1
        if test_provider("gemini"):
            success_count += 1
    
    if has_openai:
        total_tests += 1
        if test_provider("openai"):
            success_count += 1
    
    # Test auto-detection
    print(f"\n🔍 Testing Auto-Detection")
    print("-" * 40)
    
    try:
        provider = get_llm_provider()
        provider_name = type(provider).__name__.replace('Provider', '')
        print(f"✅ Auto-detected provider: {provider_name}")
        success_count += 0.5  # Half point for auto-detection
    except Exception as e:
        print(f"❌ Auto-detection failed: {e}")
    
    # Summary
    print(f"\n📊 Provider Test Summary")
    print("=" * 50)
    print(f"Tests passed: {success_count}/{total_tests + 0.5}")
    
    if success_count >= total_tests:
        print("🎉 All provider tests passed!")
        return True
    else:
        print("⚠️  Some provider tests failed. Check your API key configuration.")
        return False


# =============================================================================
# GAME MATCHING TESTS (from test_improved_matching.py)
# =============================================================================

def test_title_matching_improvements():
    """Test that the improved descriptions work better for the mentioned cases."""
    
    print("\n🧪 Testing Improved Title Matching")
    print("=" * 50)
    
    # Check for API keys
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not has_gemini and not has_openai:
        print("❌ No API key found. Add GEMINI_API_KEY or OPENAI_API_KEY to .env file")
        return False
    
    try:
        # Initialize RAG system
        print("Initializing RAG system...")
        rag = BoardGameRAG()
        
        # Use a small subset for testing if full dataset is too large
        csv_path = os.path.join(os.path.dirname(__file__), "data", "bgg_dataset.csv")
        rag.initialize(csv_path)
        
        print("✅ RAG system initialized!")
        print()
        
        # Test cases that were problematic
        test_queries = [
            {
                "query": "Recommend some highly-rated euro games with medium complexity",
                "should_not_contain": ["Euro", "European"],  # Should not match games with "Euro" in title
                "description": "Should find European-style strategy games, not games with 'Euro' in title"
            },
            {
                "query": "what is the high ranked game",
                "should_not_contain": ["High Rise", "High Heaven", "High"],  # Should not match "High" in title
                "description": "Should find highly-ranked games, not games with 'High' in title"
            },
            {
                "query": "I want a game about Mars",
                "should_contain": ["Mars"],  # SHOULD match thematic content about Mars
                "description": "Should find games about Mars theme (like Terraforming Mars)"
            },
            {
                "query": "Show me strategy games for 4 players",
                "should_not_contain": ["Strategy"],  # Should not rely on "Strategy" in title
                "description": "Should find strategy games by mechanics/category, not title words"
            },
            {
                "query": "Quick party games for large groups",
                "should_not_contain": ["Quick", "Party"],  # Should not match title words
                "description": "Should find party games by category and mechanics, not title words"
            }
        ]
        
        passed_tests = 0
        total_test_cases = len(test_queries)
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"🔍 Test {i}: {test_case['query']}")
            print(f"📝 Expected: {test_case['description']}")
            print("-" * 40)
            
            # Get search results without AI response (faster for testing)
            results = rag.search_games(test_case['query'], n_results=5)
            
            print("Top 5 results:")
            for j, result in enumerate(results, 1):
                game_name = result['metadata']['name']
                score = result['score']
                print(f"  {j}. {game_name} (similarity: {score:.3f})")
            
            test_passed = True
            
            # Check for problematic matches
            if 'should_not_contain' in test_case:
                problematic_games = []
                for result in results:
                    game_name = result['metadata']['name']
                    for avoid_word in test_case['should_not_contain']:
                        if avoid_word.lower() in game_name.lower():
                            problematic_games.append(f"{game_name} (contains '{avoid_word}')")
                
                if problematic_games:
                    print(f"⚠️  Found problematic matches:")
                    for game in problematic_games:
                        print(f"     - {game}")
                    test_passed = False
                else:
                    print("✅ No problematic title matches found!")
            
            # Check for expected matches
            if 'should_contain' in test_case:
                found_thematic = []
                for result in results:
                    game_name = result['metadata']['name']
                    for theme_word in test_case['should_contain']:
                        if theme_word.lower() in game_name.lower():
                            found_thematic.append(f"{game_name} (contains '{theme_word}')")
                
                if found_thematic:
                    print("✅ Found expected thematic matches:")
                    for game in found_thematic:
                        print(f"     - {game}")
                else:
                    print("⚠️  Did not find expected thematic matches")
                    test_passed = False
            
            if test_passed:
                passed_tests += 1
                print("✅ Test passed!")
            else:
                print("❌ Test failed!")
            
            print("\n" + "=" * 50 + "\n")
        
        print(f"🎯 Matching Test Summary: {passed_tests}/{total_test_cases} tests passed")
        
        if passed_tests == total_test_cases:
            print("🎉 All matching tests passed!")
            return True
        else:
            print("⚠️  Some matching tests failed, but improvements may still be working.")
            return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# DESCRIPTION EXAMPLES (from test_description_example.py)
# =============================================================================

def show_description_examples():
    """Show examples of the improved game descriptions."""
    
    print("\n📝 Game Description Examples")
    print("=" * 60)
    
    # Create a RAG instance to use the description method
    rag = BoardGameRAG()
    
    # Example game data
    example_games = [
        {
            'Name': 'High Rise',
            'Mechanics': 'Auction/Bidding, Building Construction, Hand Management',
            'Domains': 'Strategy Games',
            'Min Players': 1,
            'Max Players': 4,
            'Play Time': 45,
            'Min Age': 10,
            'Rating Average': 7.2,
            'Complexity Average': 2.8,
            'BGG Rank': 1500
        },
        {
            'Name': 'Terraforming Mars',
            'Mechanics': 'Card Drafting, Engine Building, Hand Management, Tile Placement',
            'Domains': 'Strategy Games',
            'Min Players': 1,
            'Max Players': 5,
            'Play Time': 120,
            'Min Age': 12,
            'Rating Average': 8.4,
            'Complexity Average': 3.2,
            'BGG Rank': 4
        },
        {
            'Name': 'Euro Crisis',
            'Mechanics': 'Trading, Negotiation, Economic Simulation',
            'Domains': 'Strategy Games',
            'Min Players': 3,
            'Max Players': 6,
            'Play Time': 90,
            'Min Age': 14,
            'Rating Average': 6.8,
            'Complexity Average': 3.5,
            'BGG Rank': 5000
        }
    ]
    
    for i, game_data in enumerate(example_games, 1):
        row = pd.Series(game_data)
        description = rag._create_game_description(row)
        
        print(f"Example {i}: {game_data['Name']}")
        print("-" * 40)
        print(description)
        print("\n" + "=" * 60 + "\n")
    
    print("🎯 Key Improvements:")
    print("✅ Game titles are de-emphasized (appear at the end)")
    print("✅ Focus on mechanics, player count, complexity")
    print("✅ Thematic words extracted separately (Mars from 'Terraforming Mars')")
    print("✅ Quality descriptors avoid title word matching")
    print("✅ Gameplay characteristics get more weight")
    
    print("\n🔍 How This Helps:")
    print("• 'euro games' → finds Strategy Games with European mechanics")
    print("• 'high ranked' → finds games with good BGG rankings")  
    print("• 'games about Mars' → finds Mars thematic content")
    print("• 'quick games' → finds games by play time, not title words")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all tests in sequence."""
    print("🚀 AI Board Game Suggester - Complete Test Suite")
    print("=" * 80)
    
    results = {
        'providers': False,
        'matching': False,
        'descriptions': True  # Description examples always "pass"
    }
    
    # Run provider tests
    print("\n" + "🔧 TESTING LLM PROVIDERS" + "\n")
    results['providers'] = run_provider_tests()
    
    # Show description examples
    print("\n" + "📋 DESCRIPTION EXAMPLES" + "\n")
    show_description_examples()
    
    # Run matching tests (only if providers work)
    if results['providers']:
        print("\n" + "🎯 TESTING GAME MATCHING" + "\n")
        results['matching'] = test_title_matching_improvements()
    else:
        print("\n❌ Skipping matching tests due to provider failures")
    
    # Final summary
    print("\n" + "📊 FINAL TEST SUMMARY")
    print("=" * 80)
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"Test Results:")
    print(f"  🔧 LLM Providers: {'✅ PASSED' if results['providers'] else '❌ FAILED'}")
    print(f"  📋 Descriptions: {'✅ PASSED' if results['descriptions'] else '❌ FAILED'}")
    print(f"  🎯 Game Matching: {'✅ PASSED' if results['matching'] else '❌ FAILED'}")
    
    print(f"\nOverall: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Your RAG system is working correctly.")
        print("\nNext steps:")
        print("  1. Run: python demo_rag.py")
        print("  2. Or run: python run_api.py")
    elif results['providers']:
        print("\n⚠️  System partially working. You can still use basic functionality.")
        print("\nTroubleshooting for failed tests:")
        print("  - Check that your dataset is properly formatted")
        print("  - Verify the RAG system initializes correctly")
        print("  - Make sure embeddings are being created properly")
    else:
        print("\n❌ CRITICAL FAILURES - System not ready")
        print("\nTroubleshooting:")
        print("  - Make sure your .env file exists")
        print("  - Verify your API keys are correct")
        print("  - Check your internet connection")
        print("  - Ensure required packages are installed")


if __name__ == "__main__":
    main()