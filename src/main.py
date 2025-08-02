"""
Main script for the Board Game Recommendation System.
Provides a command-line interface for users to get game recommendations.
"""

import sys
import os
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessor import BoardGameDataPreprocessor
from recommendation_engine import BoardGameRecommender


class BoardGameRecommendationSystem:
    """Main application class for the board game recommendation system."""
    
    def __init__(self):
        self.preprocessor = BoardGameDataPreprocessor()
        self.recommender = BoardGameRecommender()
        self.data_loaded = False
        
    def setup_system(self):
        """Initialize the recommendation system."""
        print("=== Board Game Recommendation System ===")
        print("Setting up system...")
        
        # Check if processed data exists
        processed_data_path = 'data/processed_games.csv'
        original_data_path = 'data/bgg_dataset.csv'
        
        if not os.path.exists(processed_data_path):
            print("Processed data not found. Processing dataset...")
            self._preprocess_data(original_data_path, processed_data_path)
        
        # Load data into recommender
        print("Loading data into recommendation engine...")
        self.recommender.load_data(processed_data_path, original_data_path)
        
        # Train similarity model
        print("Training similarity model...")
        self.recommender.train_similarity_model()
        
        self.data_loaded = True
        print("System ready!")
        
    def _preprocess_data(self, original_path, output_path):
        """Preprocess the raw data."""
        # Load and preprocess data
        df = self.preprocessor.load_data(original_path)
        processed_df = self.preprocessor.preprocess_features(df)
        
        # Save processed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        
        # Save preprocessor
        os.makedirs('models', exist_ok=True)
        self.preprocessor.save_preprocessor('models/preprocessor.pkl')
        
        print(f"Data preprocessing completed. Processed data saved to {output_path}")
    
    def get_user_preferences(self):
        """Get user preferences through command line interface."""
        print("\n=== Tell us about your preferences ===")
        preferences = {}
        
        # Number of players
        try:
            players_input = input("How many players? (or press Enter to skip): ").strip()
            if players_input:
                preferences['num_players'] = int(players_input)
        except ValueError:
            print("Invalid input for number of players, skipping...")
        
        # Duration preferences
        try:
            duration_input = input("Preferred play time in minutes (e.g., 30, 60-120, <90, >60): ").strip()
            if duration_input:
                if '-' in duration_input:
                    # Range like "60-120"
                    min_dur, max_dur = duration_input.split('-')
                    preferences['duration_min'] = int(min_dur.strip())
                    preferences['duration_max'] = int(max_dur.strip())
                elif duration_input.startswith('<'):
                    # Max duration like "<90"
                    preferences['duration_max'] = int(duration_input[1:])
                elif duration_input.startswith('>'):
                    # Min duration like ">60"
                    preferences['duration_min'] = int(duration_input[1:])
                else:
                    # Exact duration - use as max with some tolerance
                    target_duration = int(duration_input)
                    preferences['duration_min'] = max(0, target_duration - 30)
                    preferences['duration_max'] = target_duration + 30
        except ValueError:
            print("Invalid input for duration, skipping...")
        
        # Complexity preferences
        try:
            complexity_input = input("Preferred complexity (1-5, or range like 2-3.5): ").strip()
            if complexity_input:
                if '-' in complexity_input:
                    min_comp, max_comp = complexity_input.split('-')
                    preferences['complexity_min'] = float(min_comp.strip())
                    preferences['complexity_max'] = float(max_comp.strip())
                else:
                    target_complexity = float(complexity_input)
                    preferences['complexity_min'] = max(1.0, target_complexity - 0.5)
                    preferences['complexity_max'] = min(5.0, target_complexity + 0.5)
        except ValueError:
            print("Invalid input for complexity, skipping...")
        
        # Minimum rating
        try:
            rating_input = input("Minimum BGG rating (e.g., 7.0): ").strip()
            if rating_input:
                preferences['min_rating'] = float(rating_input)
        except ValueError:
            print("Invalid input for rating, skipping...")
        
        # Maximum age
        try:
            age_input = input("Maximum age recommendation (e.g., 12): ").strip()
            if age_input:
                preferences['max_age'] = int(age_input)
        except ValueError:
            print("Invalid input for age, skipping...")
        
        # Mechanics preferences
        print("\\nPopular mechanics:", ', '.join(self.recommender.get_popular_mechanics(15)))
        mechanics_input = input("Preferred mechanics (comma-separated, or Enter to skip): ").strip()
        if mechanics_input:
            preferences['mechanics'] = [m.strip() for m in mechanics_input.split(',')]
        
        # Domains preferences  
        print("\\nAvailable domains:", ', '.join(self.recommender.get_popular_domains()))
        domains_input = input("Preferred domains (comma-separated, or Enter to skip): ").strip()
        if domains_input:
            preferences['domains'] = [d.strip() for d in domains_input.split(',')]
            
        return preferences
    
    def display_recommendations(self, recommendations):
        """Display recommendations in a formatted way."""
        if recommendations.empty:
            print("\\nNo games found matching your criteria. Try relaxing some constraints.")
            return
            
        print(f"\\n=== Found {len(recommendations)} Recommendations ===")
        
        for idx, (_, game) in enumerate(recommendations.iterrows(), 1):
            print(f"\\n{idx}. {game['Name']} ({game.get('Year Published', 'N/A')})")
            print(f"   Players: {game.get('Min Players', '?')}-{game.get('Max Players', '?')}")
            print(f"   Time: {game.get('Play Time', '?')} minutes")
            print(f"   Age: {game.get('Min Age', '?')}+")
            print(f"   Rating: {game.get('Rating Average', 'N/A'):.1f}" + 
                  (f" (Rank #{int(game['BGG Rank'])})" if 'BGG Rank' in game and pd.notna(game['BGG Rank']) else ""))
            print(f"   Complexity: {game.get('Complexity Average', 'N/A'):.1f}/5")
            
            # Show some mechanics
            if 'Mechanics' in game and pd.notna(game['Mechanics']):
                mechanics = str(game['Mechanics'])[:100]
                if len(str(game['Mechanics'])) > 100:
                    mechanics += "..."
                print(f"   Mechanics: {mechanics}")
    
    def run_similarity_search(self):
        """Run similarity-based recommendations."""
        print("\\n=== Find Similar Games ===")
        game_name = input("Enter a game name to find similar games: ").strip()
        
        if not game_name:
            return
            
        similar_games = self.recommender.get_similar_games(game_name, n_recommendations=10)
        
        if similar_games.empty:
            print(f"No game found matching '{game_name}'. Try a different name or partial match.")
            return
            
        print(f"\\nGames similar to your search:")
        self.display_recommendations(similar_games)
    
    def run_interactive_mode(self):
        """Run the interactive recommendation system."""
        if not self.data_loaded:
            print("System not initialized. Please run setup first.")
            return
            
        while True:
            print("\\n" + "="*50)
            print("Board Game Recommendation System")
            print("="*50)
            print("1. Get personalized recommendations")
            print("2. Find similar games")
            print("3. Show popular mechanics")
            print("4. Show available domains")
            print("5. Exit")
            
            choice = input("\\nSelect an option (1-5): ").strip()
            
            if choice == '1':
                preferences = self.get_user_preferences()
                print("\\nSearching for games...")
                
                recommendations = self.recommender.recommend_games(
                    n_recommendations=10,
                    **preferences
                )
                
                self.display_recommendations(recommendations)
                
            elif choice == '2':
                self.run_similarity_search()
                
            elif choice == '3':
                mechanics = self.recommender.get_popular_mechanics(20)
                print("\\nMost popular game mechanics:")
                for i, mechanic in enumerate(mechanics, 1):
                    print(f"{i:2d}. {mechanic}")
                    
            elif choice == '4':
                domains = self.recommender.get_popular_domains()
                print("\\nAvailable game domains:")
                for i, domain in enumerate(domains, 1):
                    print(f"{i}. {domain}")
                    
            elif choice == '5':
                print("Thank you for using the Board Game Recommendation System!")
                break
                
            else:
                print("Invalid choice. Please select 1-5.")


def main():
    """Main function to run the application."""
    import pandas as pd
    
    # Initialize system
    system = BoardGameRecommendationSystem()
    
    try:
        # Setup the system
        system.setup_system()
        
        # Run interactive mode
        system.run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\\n\\nExiting... Thank you for using the Board Game Recommendation System!")
    except Exception as e:
        print(f"\\nAn error occurred: {e}")
        print("Please make sure you have the required dependencies installed and the dataset file exists.")


if __name__ == "__main__":
    main()