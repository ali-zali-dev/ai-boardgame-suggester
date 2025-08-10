#!/usr/bin/env python3
"""
Comprehensive test script for the Board Game Recommendation API.
Tests natural language parsing with both English and Persian queries.

Features:
- Fallback parsing tests (no API key required)
- Multilingual Gemini tests (English and Persian)
- Edge case and performance testing
- Individual query testing function

Usage:
    python test_api.py                    # Run all tests
    python test_interactive.py           # Interactive testing
    
Requirements:
- Set GEMINI_API_KEY environment variable for full Gemini testing
- Without API key, only fallback parsing will be tested
"""

import sys
import os
sys.path.append('src')

def test_fallback_parsing():
    """Test the fallback natural language parsing."""
    from api import parse_natural_language_query_fallback
    
    test_queries = [
        "strategic easy game for 3 players",
        "quick party game for 6 people",
        "family game with dice rolling",
        "complex strategy game"
    ]
    
    print("Testing fallback parsing (without Gemini):")
    print("=" * 50)
    
    for query in test_queries:
        filters, interpretation = parse_natural_language_query_fallback(query)
        print(f"\nQuery: '{query}'")
        print(f"Interpretation: {interpretation}")
        print(f"Filters: {filters}")
    
    print("\n" + "=" * 50)
    print("Fallback parsing test completed!")


def test_multilingual_queries():
    """Test comprehensive English and Persian queries for Gemini."""
    from api import parse_natural_language_query_with_gemini, gemini_client
    
    # Enhanced English test queries
    english_queries = [
        # Basic recommendations
        "I want a strategy game for 4 players",
        "Recommend a quick party game for 6-8 people",
        "Find me a complex eurogame that takes 2-3 hours",
        "Show me family-friendly games under 60 minutes",
        
        # Specific mechanics and themes
        "I love worker placement games with medium complexity",
        "Find cooperative games that are good for beginners",
        "I want deck building games with high ratings",
        "Show me abstract strategy games for 2 players",
        
        # Complex queries
        "I need a game that's easy to learn but hard to master, for 3-5 players, around 90 minutes",
        "Find me the best rated games with dice rolling mechanics that work well with 4 players",
        "I want thematic games with storytelling elements, not too complex, for adults",
        "Show me gateway games that are perfect for introducing people to modern board gaming"
    ]
    
    # Persian test queries (Farsi)
    persian_queries = [
        # Basic recommendations in Persian
        "یک بازی استراتژیک برای ۴ نفر می‌خوام",
        "بازی سریع و مهمانی برای ۶ نفر پیشنهاد بده",
        "بازی خانوادگی آسان زیر ۶۰ دقیقه نشون بده",
        "بازی پیچیده و سنگین برای ۲ نفر می‌خوام",
        
        # More complex Persian queries
        "بازی‌های تعاونی که برای مبتدی‌ها خوب باشه پیدا کن",
        "بهترین بازی‌های امتیاز بالا رو نشون بده",
        "بازی با تاس که برای ۳ نفر مناسب باشه می‌خوام",
        "بازی‌های کارت‌سازی با پیچیدگی متوسط پیشنهاد بده",
        
        # Mixed complexity
        "بازی که یادگیری آسان ولی تسلط دشوار داشته باشه، برای ۳-۵ نفر",
        "بازی‌های مدرن که برای آشنایی با بازی‌های رومیزی عالی باشن"
    ]
    
    if not gemini_client:
        print("⚠️  Gemini client not available. Skipping multilingual tests.")
        print("Set GEMINI_API_KEY environment variable to test Gemini integration.")
        return
    
    print("Testing multilingual queries with Gemini:")
    print("=" * 60)
    
    # Test English queries
    print("\n🇺🇸 ENGLISH QUERIES:")
    print("-" * 40)
    for i, query in enumerate(english_queries, 1):
        try:
            filters, interpretation = parse_natural_language_query_with_gemini(query)
            print(f"\n{i}. Query: '{query}'")
            print(f"   Interpretation: {interpretation}")
            print(f"   Filters: {filters}")
        except Exception as e:
            print(f"\n{i}. Query: '{query}'")
            print(f"   ❌ Error: {e}")
    
    # Test Persian queries
    print("\n\n🇮🇷 PERSIAN QUERIES (فارسی):")
    print("-" * 40)
    for i, query in enumerate(persian_queries, 1):
        try:
            filters, interpretation = parse_natural_language_query_with_gemini(query)
            print(f"\n{i}. Query: '{query}'")
            print(f"   Interpretation: {interpretation}")
            print(f"   Filters: {filters}")
        except Exception as e:
            print(f"\n{i}. Query: '{query}'")
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Multilingual testing completed!")


def test_gemini_performance():
    """Test Gemini's performance with edge cases and complex scenarios."""
    from api import parse_natural_language_query_with_gemini, gemini_client
    
    if not gemini_client:
        print("⚠️  Gemini client not available. Skipping performance tests.")
        return
    
    edge_cases = [
        # Ambiguous queries
        "something fun",
        "good game",
        "بازی خوب", # "good game" in Persian
        
        # Very specific queries
        "I want a 2-player abstract strategy game with no luck, high complexity, exactly 60 minutes duration, rated above 8.0",
        "بازی استراتژیک دو نفره بدون شانس، پیچیدگی بالا، دقیقاً ۶۰ دقیقه، امتیاز بالای ۸", # Same in Persian
        
        # Mixed language
        "strategy بازی for دو نفر players",
        
        # Contradictory requirements
        "easy complex game for beginners with expert-level strategy",
        
        # Non-game related query
        "What's the weather like today?",
        "امروز هوا چطوره؟" # "How's the weather today?" in Persian
    ]
    
    print("Testing Gemini performance with edge cases:")
    print("=" * 50)
    
    for i, query in enumerate(edge_cases, 1):
        try:
            filters, interpretation = parse_natural_language_query_with_gemini(query)
            print(f"\n{i}. Query: '{query}'")
            print(f"   Interpretation: {interpretation}")
            print(f"   Filters: {filters}")
        except Exception as e:
            print(f"\n{i}. Query: '{query}'")
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Performance testing completed!")

def test_single_query(query: str):
    """Test a single query with both Gemini and fallback parsing."""
    from api import parse_natural_language_query_with_gemini, parse_natural_language_query_fallback, gemini_client
    
    print(f"Testing query: '{query}'")
    print("-" * 50)
    
    # Test with Gemini if available
    if gemini_client:
        try:
            filters, interpretation = parse_natural_language_query_with_gemini(query)
            print("🤖 Gemini Result:")
            print(f"   Interpretation: {interpretation}")
            print(f"   Filters: {filters}")
        except Exception as e:
            print(f"🤖 Gemini Error: {e}")
    else:
        print("🤖 Gemini: Not available (no API key)")
    
    # Test with fallback
    try:
        filters, interpretation = parse_natural_language_query_fallback(query)
        print("🔧 Fallback Result:")
        print(f"   Interpretation: {interpretation}")
        print(f"   Filters: {filters}")
    except Exception as e:
        print(f"🔧 Fallback Error: {e}")
    
    print("-" * 50)


def test_api_imports():
    """Test if the API can be imported without errors."""
    try:
        from api import app, gemini_client
        print("✅ API imported successfully!")
        
        if gemini_client:
            print("✅ Gemini client initialized!")
        else:
            print("⚠️  Gemini client not initialized (API key not set)")
            
        return True
    except Exception as e:
        print(f"❌ Error importing API: {e}")
        return False

if __name__ == "__main__":
    print("🎲 Board Game Recommendation API Test Suite")
    print("=" * 60)
    
    # Test imports
    if test_api_imports():
        print()
        
        # Test fallback parsing (always works)
        test_fallback_parsing()
        print()
        
        # Test multilingual queries with Gemini
        test_multilingual_queries()
        print()
        
        # Test edge cases and performance
        test_gemini_performance()
        
    else:
        print("Cannot run tests due to import errors.")
        sys.exit(1)
    
    print("\n🎯 All tests completed!")
    print("=" * 60)