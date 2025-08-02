"""
Board game recommendation engine.
Implements filtering-based recommendations based on user preferences.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import pickle


class BoardGameRecommender:
    """Filtering-based board game recommendation system."""
    
    def __init__(self):
        self.games_data = None
        self.processed_data = None
        self.feature_columns = None
        self.nn_model = None
        
    def load_data(self, processed_csv_path, original_csv_path=None):
        """Load processed and original game data."""
        print("Loading game data...")
        
        # Load processed data for similarity calculations
        self.processed_data = pd.read_csv(processed_csv_path)
        
        # Load original data for display purposes
        if original_csv_path:
            self.games_data = pd.read_csv(original_csv_path, sep=';', encoding='utf-8')
            # Clean numeric columns
            numeric_cols = ['Rating Average', 'Complexity Average']
            for col in numeric_cols:
                if col in self.games_data.columns:
                    self.games_data[col] = self.games_data[col].astype(str).str.replace(',', '.').astype(float)
        else:
            self.games_data = self.processed_data
            
        # Define feature columns for similarity calculation
        self.feature_columns = [col for col in self.processed_data.columns 
                               if col not in ['ID', 'Name', 'Year Published', 'Min Players', 'Max Players', 
                                            'Play Time', 'Min Age', 'Users Rated', 'Rating Average', 
                                            'BGG Rank', 'Complexity Average', 'Owned Users', 'Mechanics', 'Domains']]
        
        print(f"Loaded {len(self.games_data)} games")
        print(f"Feature columns for similarity: {len(self.feature_columns)}")
        
    def train_similarity_model(self):
        """Train a k-nearest neighbors model for similarity-based recommendations."""
        print("Training similarity model...")
        
        if self.feature_columns is None:
            raise ValueError("Must load data first")
            
        # Use feature columns for similarity
        feature_data = self.processed_data[self.feature_columns].fillna(0)
        
        # Train KNN model
        self.nn_model = NearestNeighbors(n_neighbors=50, metric='cosine', algorithm='brute')
        self.nn_model.fit(feature_data)
        
        print("Similarity model trained successfully")
    
    def filter_games(self, num_players=None, duration_min=None, duration_max=None, 
                    complexity_min=None, complexity_max=None, min_rating=None, 
                    max_age=None, mechanics=None, domains=None):
        """
        Filter games based on user criteria.
        
        Args:
            num_players: Desired number of players
            duration_min: Minimum play time in minutes
            duration_max: Maximum play time in minutes
            complexity_min: Minimum complexity (1-5 scale)
            complexity_max: Maximum complexity (1-5 scale)
            min_rating: Minimum BGG rating
            max_age: Maximum recommended age
            mechanics: List of desired mechanics
            domains: List of desired domains
            
        Returns:
            Filtered DataFrame of games
        """
        print(f"Filtering games with criteria...")
        
        df = self.games_data.copy()
        original_count = len(df)
        
        # Filter by number of players
        if num_players is not None:
            df = df[(df['Min Players'] <= num_players) & (df['Max Players'] >= num_players)]
            print(f"  After player count filter ({num_players}): {len(df)} games")
        
        # Filter by duration
        if duration_min is not None:
            df = df[df['Play Time'] >= duration_min]
            print(f"  After min duration filter ({duration_min}min): {len(df)} games")
            
        if duration_max is not None:
            df = df[df['Play Time'] <= duration_max]
            print(f"  After max duration filter ({duration_max}min): {len(df)} games")
        
        # Filter by complexity
        if complexity_min is not None:
            df = df[df['Complexity Average'] >= complexity_min]
            print(f"  After min complexity filter ({complexity_min}): {len(df)} games")
            
        if complexity_max is not None:
            df = df[df['Complexity Average'] <= complexity_max]
            print(f"  After max complexity filter ({complexity_max}): {len(df)} games")
        
        # Filter by rating
        if min_rating is not None:
            df = df[df['Rating Average'] >= min_rating]
            print(f"  After rating filter ({min_rating}+): {len(df)} games")
        
        # Filter by age
        if max_age is not None:
            df = df[df['Min Age'] <= max_age]
            print(f"  After age filter (≤{max_age}): {len(df)} games")
        
        # Filter by mechanics
        if mechanics is not None and len(mechanics) > 0:
            mechanics_filter = df['Mechanics'].str.contains('|'.join(mechanics), case=False, na=False)
            df = df[mechanics_filter]
            print(f"  After mechanics filter: {len(df)} games")
        
        # Filter by domains
        if domains is not None and len(domains) > 0:
            domains_filter = df['Domains'].str.contains('|'.join(domains), case=False, na=False)
            df = df[domains_filter]
            print(f"  After domains filter: {len(df)} games")
        
        print(f"Final filtered results: {len(df)} games (from {original_count})")
        return df
    
    def get_similar_games(self, game_name, n_recommendations=10):
        """Get games similar to a specific game."""
        if self.nn_model is None:
            raise ValueError("Must train similarity model first")
            
        # Find the game
        game_matches = self.games_data[self.games_data['Name'].str.contains(game_name, case=False, na=False)]
        
        if len(game_matches) == 0:
            print(f"No game found matching '{game_name}'")
            return pd.DataFrame()
        
        # Use the first match
        target_game = game_matches.iloc[0]
        game_idx = target_game.name
        
        print(f"Finding games similar to: {target_game['Name']}")
        
        # Get feature vector for the target game
        feature_vector = self.processed_data.iloc[game_idx][self.feature_columns].fillna(0).values.reshape(1, -1)
        
        # Find similar games
        distances, indices = self.nn_model.kneighbors(feature_vector, n_neighbors=n_recommendations+1)
        
        # Get similar games (excluding the target game itself)
        similar_indices = indices[0][1:]  # Skip first result (the game itself)
        similar_games = self.games_data.iloc[similar_indices].copy()
        similar_games['Similarity_Score'] = 1 - distances[0][1:]  # Convert distance to similarity
        
        return similar_games
    
    def recommend_games(self, num_players=None, duration_min=None, duration_max=None,
                       complexity_min=None, complexity_max=None, min_rating=None,
                       max_age=None, mechanics=None, domains=None, n_recommendations=10,
                       sort_by='Rating Average'):
        """
        Get game recommendations based on filtering criteria.
        
        Args:
            num_players: Desired number of players
            duration_min: Minimum play time in minutes
            duration_max: Maximum play time in minutes  
            complexity_min: Minimum complexity (1-5 scale)
            complexity_max: Maximum complexity (1-5 scale)
            min_rating: Minimum BGG rating
            max_age: Maximum recommended age
            mechanics: List of desired mechanics
            domains: List of desired domains
            n_recommendations: Number of recommendations to return
            sort_by: Column to sort recommendations by
            
        Returns:
            DataFrame of recommended games
        """
        
        # Filter games based on criteria
        filtered_games = self.filter_games(
            num_players=num_players,
            duration_min=duration_min, 
            duration_max=duration_max,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
            min_rating=min_rating,
            max_age=max_age,
            mechanics=mechanics,
            domains=domains
        )
        
        if len(filtered_games) == 0:
            print("No games match your criteria!")
            return pd.DataFrame()
        
        # Sort and return top recommendations
        if sort_by in filtered_games.columns:
            filtered_games = filtered_games.sort_values(by=sort_by, ascending=False)
        
        recommendations = filtered_games.head(n_recommendations)
        
        # Select relevant columns for display
        display_columns = ['Name', 'Year Published', 'Min Players', 'Max Players', 
                          'Play Time', 'Min Age', 'Rating Average', 'Complexity Average',
                          'BGG Rank', 'Mechanics', 'Domains']
        
        available_columns = [col for col in display_columns if col in recommendations.columns]
        
        return recommendations[available_columns]
    
    def get_popular_mechanics(self, top_n=20):
        """Get the most popular game mechanics."""
        if self.games_data is None:
            return []
            
        all_mechanics = []
        for mechanics_str in self.games_data['Mechanics'].fillna(''):
            if mechanics_str:
                mechanics_list = [m.strip() for m in str(mechanics_str).split(',')]
                all_mechanics.extend(mechanics_list)
        
        from collections import Counter
        mechanics_counter = Counter(all_mechanics)
        return [mechanic for mechanic, count in mechanics_counter.most_common(top_n)]
    
    def get_popular_domains(self, top_n=10):
        """Get the most popular game domains."""
        if self.games_data is None:
            return []
            
        all_domains = []
        for domains_str in self.games_data['Domains'].fillna(''):
            if domains_str:
                domains_list = [d.strip() for d in str(domains_str).split(',')]
                all_domains.extend(domains_list)
        
        from collections import Counter
        domains_counter = Counter(all_domains)
        return [domain for domain, count in domains_counter.most_common(top_n)]


if __name__ == "__main__":
    # Test the recommender
    recommender = BoardGameRecommender()
    
    # Load data
    recommender.load_data('../data/processed_games.csv', '../data/bgg_dataset.csv')
    
    # Train similarity model
    recommender.train_similarity_model()
    
    # Test filtering
    print("\n=== Testing Filtering ===")
    recommendations = recommender.recommend_games(
        num_players=4,
        duration_max=90,
        complexity_min=2.0,
        complexity_max=3.5,
        min_rating=7.0,
        n_recommendations=5
    )
    
    print("\nTop 5 recommendations:")
    print(recommendations[['Name', 'Rating Average', 'Complexity Average', 'Play Time']].to_string())
    
    # Test similarity
    print("\n=== Testing Similarity ===")
    similar_games = recommender.get_similar_games("Catan", n_recommendations=5)
    if not similar_games.empty:
        print(f"\nGames similar to Catan:")
        print(similar_games[['Name', 'Rating Average', 'Similarity_Score']].to_string())