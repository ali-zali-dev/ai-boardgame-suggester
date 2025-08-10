"""
Demo script for the Board Game Recommendation System.
Shows example usage without requiring user interaction.
"""

import sys
import os

# Add src to path
sys.path.append('src')

from src.data_preprocessor import BoardGameDataPreprocessor
from src.recommendation_engine import BoardGameRecommender


def run_demo():
    """Run a demonstration of the recommendation system."""
    print("=== Board Game Recommendation System Demo ===")
    
    # Check if processed data exists
    processed_data_path = 'data/processed_games.csv'
    original_data_path = 'data/bgg_dataset.csv'
    
    if not os.path.exists(original_data_path):
        print(f"Error: Dataset not found at {original_data_path}")
        print("Please ensure the BGG dataset is in the data/ directory.")
        return
    
    # Set up system if needed
    if not os.path.exists(processed_data_path):
        print("Setting up system for first time...")
        preprocessor = BoardGameDataPreprocessor()
        df = preprocessor.load_data(original_data_path)
        processed_df = preprocessor.preprocess_features(df)
        processed_df.to_csv(processed_data_path, index=False)
        
        os.makedirs('models', exist_ok=True)
        preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    # Initialize recommender
    print("\nInitializing recommendation engine...")
    recommender = BoardGameRecommender()
    recommender.load_data(processed_data_path, original_data_path)
    recommender.train_similarity_model()
    
    # Demo 1: Family game recommendations
    print("\n" + "="*60)
    print("DEMO 1: Family Game Night (4 players, 60-90 min, not too complex)")
    print("="*60)
    
    family_recommendations = recommender.recommend_games(
        num_players=4,
        duration_min=60,
        duration_max=90,
        complexity_max=2.5,
        min_rating=7.0,
        n_recommendations=5
    )
    
    display_recommendations(family_recommendations, "Family Games")
    
    # Demo 2: Strategy game recommendations
    print("\n" + "="*60)
    print("DEMO 2: Strategy Enthusiasts (2-3 players, complex games, 90+ min)")
    print("="*60)
    
    strategy_recommendations = recommender.recommend_games(
        num_players=3,
        duration_min=90,
        complexity_min=3.0,
        min_rating=7.5,
        domains=['Strategy Games'],
        n_recommendations=5
    )
    
    display_recommendations(strategy_recommendations, "Strategy Games")
    
    # Demo 3: Quick games
    print("\n" + "="*60)
    print("DEMO 3: Quick Games (any player count, under 45 min)")
    print("="*60)
    
    quick_recommendations = recommender.recommend_games(
        duration_max=45,
        min_rating=7.0,
        n_recommendations=5
    )
    
    display_recommendations(quick_recommendations, "Quick Games")
    
    # Demo 4: Similarity search
    print("\n" + "="*60)
    print("DEMO 4: Games Similar to Popular Titles")
    print("="*60)
    
    # Try to find games similar to Catan
    similar_games = recommender.get_similar_games("Catan", n_recommendations=5)
    if not similar_games.empty:
        print("\nGames similar to Catan:")
        display_recommendations(similar_games, "Similar to Catan", show_similarity=True)
    
    # Try games similar to Wingspan
    similar_games2 = recommender.get_similar_games("Wingspan", n_recommendations=3)
    if not similar_games2.empty:
        print("\nGames similar to Wingspan:")
        display_recommendations(similar_games2, "Similar to Wingspan", show_similarity=True)
    
    # Show popular mechanics and domains
    print("\n" + "="*60)
    print("DEMO 5: Popular Mechanics and Domains")
    print("="*60)
    
    print("\nMost Popular Game Mechanics:")
    mechanics = recommender.get_popular_mechanics(10)
    for i, mechanic in enumerate(mechanics, 1):
        print(f"  {i:2d}. {mechanic}")
    
    print("\nAvailable Game Domains:")
    domains = recommender.get_popular_domains()
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")
    
    print("\n" + "="*60)
    print("Demo completed! Use the API at http://localhost:8000 for recommendations.")
    print("Run 'python run_api.py' to start the API server.")
    print("="*60)


def display_recommendations(recommendations, title, show_similarity=False):
    """Display recommendations in a formatted way."""
    import pandas as pd
    
    if recommendations.empty:
        print(f"\nNo {title.lower()} found matching the criteria.")
        return
        
    print(f"\nTop {title}:")
    print("-" * 40)
    
    for idx, (_, game) in enumerate(recommendations.iterrows(), 1):
        print(f"\n{idx}. {game['Name']} ({game.get('Year Published', 'N/A')})")
        print(f"   Players: {game.get('Min Players', '?')}-{game.get('Max Players', '?')}")
        print(f"   Time: {game.get('Play Time', '?')} minutes")
        print(f"   Age: {game.get('Min Age', '?')}+")
        
        rating_str = f"   Rating: {game.get('Rating Average', 'N/A'):.1f}"
        if 'BGG Rank' in game and pd.notna(game['BGG Rank']):
            rating_str += f" (Rank #{int(game['BGG Rank'])})"
        print(rating_str)
        
        print(f"   Complexity: {game.get('Complexity Average', 'N/A'):.1f}/5")
        
        if show_similarity and 'Similarity_Score' in game:
            print(f"   Similarity: {game['Similarity_Score']:.2f}")
        
        # Show some mechanics (truncated)
        if 'Mechanics' in game and pd.notna(game['Mechanics']):
            mechanics = str(game['Mechanics'])[:80]
            if len(str(game['Mechanics'])) > 80:
                mechanics += "..."
            print(f"   Mechanics: {mechanics}")


if __name__ == "__main__":
    import pandas as pd
    run_demo()