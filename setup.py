"""
Setup script for the Board Game Recommendation System.
Handles data preprocessing and initial system setup.
"""

import os
import sys

# Add src to path
sys.path.append('src')

from src.data_preprocessor import BoardGameDataPreprocessor
from src.recommendation_engine import BoardGameRecommender


def setup_system():
    """Set up the board game recommendation system."""
    print("=== Board Game Recommendation System Setup ===")
    
    # Check if dataset exists
    dataset_path = 'data/bgg_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the BGG dataset is in the data/ directory.")
        return False
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Initialize preprocessor
    print("Initializing data preprocessor...")
    preprocessor = BoardGameDataPreprocessor()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = preprocessor.load_data(dataset_path)
    processed_df = preprocessor.preprocess_features(df)
    
    # Save processed data
    processed_path = 'data/processed_games.csv'
    processed_df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")
    
    # Save preprocessor model
    preprocessor_path = 'models/preprocessor.pkl'
    preprocessor.save_preprocessor(preprocessor_path)
    print(f"Preprocessor model saved to {preprocessor_path}")
    
    # Test the recommendation engine
    print("Testing recommendation engine...")
    recommender = BoardGameRecommender()
    recommender.load_data(processed_path, dataset_path)
    recommender.train_similarity_model()
    
    # Quick test
    test_recommendations = recommender.recommend_games(
        num_players=4,
        duration_max=90,
        min_rating=7.0,
        n_recommendations=3
    )
    
    print(f"Test successful! Found {len(test_recommendations)} recommendations.")
    if not test_recommendations.empty:
        print("Sample recommendation:")
        first_game = test_recommendations.iloc[0]
        print(f"  - {first_game['Name']} (Rating: {first_game['Rating Average']:.1f})")
    
    print("=== Setup Complete! ===")
    print("You can now run: python main.py")
    print("Or try the demo: python demo.py")
    
    return True


if __name__ == "__main__":
    success = setup_system()
    if not success:
        sys.exit(1)