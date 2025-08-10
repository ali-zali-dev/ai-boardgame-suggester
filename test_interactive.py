#!/usr/bin/env python3
"""
Interactive test script for testing individual queries.
Useful for quickly testing specific English or Persian queries with Gemini.
"""

import sys
import os
sys.path.append('src')

from test_api import test_single_query

def interactive_test():
    """Interactive testing interface."""
    print("🎲 Interactive Board Game Query Tester")
    print("=" * 50)
    print("Enter queries in English or Persian to test with Gemini and fallback parsing.")
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("Type 'examples' to see sample queries.")
    print()
    
    while True:
        try:
            query = input("📝 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query.lower() == 'examples':
                show_examples()
                continue
                
            if not query:
                print("Please enter a query or 'quit' to exit.")
                continue
            
            print()
            test_single_query(query)
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("👋 Goodbye!")

def show_examples():
    """Show example queries in both languages."""
    print("\n📚 Example Queries:")
    print("-" * 30)
    
    print("\n🇺🇸 English Examples:")
    english_examples = [
        "strategy game for 4 players",
        "quick party game for 6 people", 
        "cooperative game for beginners",
        "complex eurogame 2-3 hours",
        "deck building game high rating"
    ]
    for ex in english_examples:
        print(f"   • {ex}")
    
    print("\n🇮🇷 Persian Examples (فارسی):")
    persian_examples = [
        "بازی استراتژیک برای ۴ نفر",
        "بازی سریع مهمانی برای ۶ نفر",
        "بازی تعاونی برای مبتدی‌ها", 
        "بازی پیچیده ۲-۳ ساعته",
        "بازی کارت‌سازی امتیاز بالا"
    ]
    for ex in persian_examples:
        print(f"   • {ex}")
    print()

if __name__ == "__main__":
    interactive_test()