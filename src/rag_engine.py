"""
RAG (Retrieval-Augmented Generation) engine for board game recommendations.
Uses ChromaDB for vector storage and supports multiple LLM providers (OpenAI, Gemini).
"""

import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any, Optional
import json
import re
from dotenv import load_dotenv
from .llm_providers import get_llm_provider, LLMProvider

# Load environment variables
load_dotenv()


class BoardGameRAG:
    """RAG system for board game recommendations and information retrieval."""
    
    def __init__(self, provider_name: Optional[str] = None):
        """
        Initialize the RAG system.
        
        Args:
            provider_name: LLM provider to use ("openai", "gemini", or None for auto-detect)
        """
        # Initialize LLM provider
        try:
            self.llm_provider = get_llm_provider(provider_name)
            print(f"✅ LLM Provider initialized: {type(self.llm_provider).__name__}")
        except ValueError as e:
            raise ValueError(f"Failed to initialize LLM provider: {e}")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection_name = "board_games"
        self.collection = None
        self.games_df = None
        
    def load_and_process_data(self, csv_path: str) -> None:
        """Load board game data from CSV and process it for RAG."""
        print("Loading board game data...")
        
        # Load CSV with proper encoding and separator
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        print(f"Loaded {len(df)} games")
        
        # Clean and process the data
        df = self._clean_data(df)
        self.games_df = df
        
        print(f"Processed {len(df)} games successfully")
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the board game data."""
        # Replace commas with dots in numeric columns (European format)
        numeric_cols = ['Rating Average', 'Complexity Average']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        # Fill missing values
        df = df.fillna({
            'Name': 'Unknown Game',
            'Year Published': 0,
            'Min Players': 1,
            'Max Players': 8,
            'Play Time': 60,
            'Min Age': 8,
            'Rating Average': 0.0,
            'Complexity Average': 0.0,
            'BGG Rank': 999999,
            'Mechanics': 'Unknown',
            'Domains': 'Unknown'
        })
        
        # Clean text fields
        text_cols = ['Name', 'Mechanics', 'Domains']
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def _create_game_description(self, row: pd.Series) -> str:
        """Create a rich text description for each game, focusing on gameplay over title words."""
        mechanics = row.get('Mechanics', '').replace(',', ', ')
        domains = row.get('Domains', '').replace(',', ', ')
        name = row.get('Name', 'Unknown')
        
        # Extract thematic words from title (words that indicate subject matter)
        thematic_keywords = self._extract_thematic_words(name)
        
        # Create gameplay-focused description
        description = f"""
        Game Type: {domains}
        Mechanics: {mechanics}
        Player Count: {row.get('Min Players', 1)}-{row.get('Max Players', 8)} players
        Duration: {row.get('Play Time', 60)} minutes
        Age Range: {row.get('Min Age', 8)}+ years
        Complexity Level: {row.get('Complexity Average', 0):.1f} out of 5
        Quality Rating: {row.get('Rating Average', 0):.1f} out of 10
        BGG Ranking: #{row.get('BGG Rank', 'Unranked')}
        
        This is a {domains.lower()} with gameplay mechanics including {mechanics.lower()}.
        The game accommodates {row.get('Min Players', 1)} to {row.get('Max Players', 8)} players.
        Playing time is approximately {row.get('Play Time', 60)} minutes.
        Complexity rating is {row.get('Complexity Average', 0):.1f} making it suitable for {'beginners' if row.get('Complexity Average', 0) < 2.5 else 'intermediate players' if row.get('Complexity Average', 0) < 3.5 else 'experienced gamers'}.
        Community rating of {row.get('Rating Average', 0):.1f} indicates {'excellent' if row.get('Rating Average', 0) >= 8.0 else 'very good' if row.get('Rating Average', 0) >= 7.0 else 'good' if row.get('Rating Average', 0) >= 6.0 else 'average'} quality.
        """
        
        # Only add thematic content if there are actual thematic words
        if thematic_keywords:
            description += f"\nThematic Elements: {', '.join(thematic_keywords)}"
        
        # Add title only at the end with lower emphasis
        description += f"\nGame Title: {name}"
        
        return description.strip()
    
    def _extract_thematic_words(self, game_name: str) -> List[str]:
        """Extract thematic/subject matter words from game titles."""
        # Common thematic words that indicate subject matter
        thematic_words = {
            # Places and settings
            'mars', 'space', 'earth', 'planet', 'galaxy', 'universe', 'medieval', 'ancient', 'modern', 'future',
            'city', 'town', 'village', 'kingdom', 'empire', 'island', 'ocean', 'sea', 'mountain', 'forest',
            'desert', 'arctic', 'tropical', 'underground', 'atlantis', 'egypt', 'rome', 'greece', 'japan',
            'china', 'america', 'europe', 'africa', 'asia', 'civilization', 'colony', 'frontier',
            
            # Themes and subjects
            'war', 'battle', 'combat', 'army', 'military', 'soldier', 'knight', 'ninja', 'samurai', 'pirate',
            'robot', 'android', 'alien', 'monster', 'dragon', 'zombie', 'vampire', 'werewolf', 'magic',
            'wizard', 'witch', 'spell', 'fantasy', 'adventure', 'quest', 'treasure', 'gold', 'diamond',
            'farming', 'agriculture', 'railroad', 'train', 'ship', 'aircraft', 'plane', 'submarine',
            'factory', 'industry', 'technology', 'science', 'research', 'exploration', 'discovery',
            
            # Activities and professions
            'cooking', 'restaurant', 'chef', 'food', 'wine', 'beer', 'coffee', 'tea', 'garden', 'flower',
            'animal', 'zoo', 'safari', 'bird', 'fish', 'cat', 'dog', 'horse', 'farm', 'ranch',
            'doctor', 'hospital', 'medicine', 'detective', 'police', 'crime', 'mystery', 'spy',
            'merchant', 'trader', 'market', 'shop', 'store', 'business', 'company', 'corporation',
            
            # Generic descriptive themes (but not quality words)
            'survival', 'escape', 'rescue', 'defense', 'attack', 'invasion', 'revolution', 'rebellion',
            'cooperation', 'collaboration', 'competition', 'racing', 'speed', 'time', 'memory',
            'puzzle', 'riddle', 'logic', 'strategy', 'tactics', 'planning', 'building', 'construction'
        }
        
        # Words to avoid (common game terms that aren't thematic)
        avoid_words = {
            'game', 'games', 'board', 'card', 'cards', 'dice', 'edition', 'expansion', 'deluxe',
            'ultimate', 'complete', 'advanced', 'basic', 'mini', 'pocket', 'travel', 'family',
            'party', 'quick', 'fast', 'slow', 'easy', 'hard', 'simple', 'complex', 'new', 'old',
            'first', 'second', 'third', 'last', 'final', 'original', 'classic', 'modern', 'traditional',
            'high', 'low', 'big', 'small', 'great', 'grand', 'super', 'mega', 'ultra', 'best',
            'top', 'bottom', 'left', 'right', 'front', 'back', 'inside', 'outside', 'upper', 'lower',
            'master', 'expert', 'professional', 'amateur', 'beginner', 'advanced', 'legendary',
            'epic', 'awesome', 'amazing', 'incredible', 'fantastic', 'wonderful', 'excellent',
            'perfect', 'supreme', 'elite', 'premium', 'standard', 'regular', 'normal', 'special'
        }
        
        words = game_name.lower().split()
        thematic = []
        
        for word in words:
            # Clean the word (remove punctuation)
            clean_word = ''.join(c for c in word if c.isalpha())
            
            # Check if it's a thematic word and not in avoid list
            if clean_word in thematic_words and clean_word not in avoid_words and len(clean_word) > 2:
                thematic.append(clean_word)
        
        return thematic
    
    def create_vector_database(self) -> None:
        """Create embeddings and populate the vector database."""
        if self.games_df is None:
            raise ValueError("No data loaded. Call load_and_process_data() first.")
        
        print("Creating vector database...")
        
        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass
        
        # Create new collection with embedding function
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Create documents and metadata
        documents = []
        metadatas = []
        ids = []
        
        for idx, row in self.games_df.iterrows():
            # Create rich description
            description = self._create_game_description(row)
            documents.append(description)
            
            # Create metadata
            metadata = {
                'game_id': str(row.get('ID', idx)),
                'name': str(row.get('Name', 'Unknown')),
                'year': int(row.get('Year Published', 0)),
                'min_players': int(row.get('Min Players', 1)),
                'max_players': int(row.get('Max Players', 8)),
                'play_time': int(row.get('Play Time', 60)),
                'min_age': int(row.get('Min Age', 8)),
                'rating': float(row.get('Rating Average', 0)),
                'complexity': float(row.get('Complexity Average', 0)),
                'rank': int(row.get('BGG Rank', 999999)),
                'mechanics': str(row.get('Mechanics', '')),
                'domains': str(row.get('Domains', ''))
            }
            metadatas.append(metadata)
            ids.append(f"game_{idx}")
        
        # Create embeddings using the LLM provider
        print("Creating embeddings...")
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            # Create embeddings for this batch
            try:
                embeddings = self.llm_provider.create_embeddings(batch_docs)
                
                # Add to collection with embeddings
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids,
                    embeddings=embeddings
                )
                
                print(f"Added batch {i//batch_size + 1}/{total_batches}")
            except Exception as e:
                print(f"Error creating embeddings for batch {i//batch_size + 1}: {e}")
                # Fallback: add without custom embeddings (ChromaDB will use default)
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
        
        print(f"Vector database created with {len(documents)} games")
    
    def search_games(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for games using semantic similarity."""
        if self.collection is None:
            raise ValueError("Vector database not created. Call create_vector_database() first.")
        
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'score': results['distances'][0][i],
                'metadata': results['metadatas'][0][i],
                'document': results['documents'][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate a natural language response using retrieved games."""
        # Create context from search results
        context = "Here are some relevant board games:\n\n"
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            context += f"{i}. **{metadata['name']}** ({metadata['year']})\n"
            context += f"   - Players: {metadata['min_players']}-{metadata['max_players']}\n"
            context += f"   - Time: {metadata['play_time']} minutes\n"
            context += f"   - Rating: {metadata['rating']:.1f}/10\n"
            context += f"   - Complexity: {metadata['complexity']:.1f}/5\n"
            context += f"   - Mechanics: {metadata['mechanics']}\n"
            context += f"   - Categories: {metadata['domains']}\n\n"
        
        # Create the prompt
        prompt = f"""You are a knowledgeable board game expert helping users find the perfect games.

User Query: {query}

{context}

Based on the games above, provide a helpful response to the user's query. Include:
1. Direct recommendations based on their request
2. Brief explanations of why these games fit their criteria
3. Any additional insights about the mechanics or gameplay
4. Suggestions for different player counts or complexity levels if relevant

Be conversational, enthusiastic, and informative. Focus on helping the user make a good choice."""
        
        # Generate response using the LLM provider
        try:
            response = self.llm_provider.generate_response(prompt, max_tokens=500, temperature=0.7)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            # Fallback response
            return f"Based on your query '{query}', I found {len(search_results)} relevant games. Here are the top recommendations: " + ", ".join([r['metadata']['name'] for r in search_results[:3]])
    
    def query(self, user_query: str, n_results: int = 5) -> Dict[str, Any]:
        """Main query method that combines search and generation."""
        # Search for relevant games
        search_results = self.search_games(user_query, n_results)
        
        # Generate response
        response = self.generate_response(user_query, search_results)
        
        return {
            'query': user_query,
            'response': response,
            'games': search_results,
            'metadata': {
                'num_results': len(search_results),
                'provider': type(self.llm_provider).__name__
            }
        }
    
    def initialize(self, csv_path: str) -> None:
        """Initialize the entire RAG system."""
        print("Initializing Board Game RAG System...")
        self.load_and_process_data(csv_path)
        self.create_vector_database()
        print("RAG system ready!")


if __name__ == "__main__":
    # Example usage
    try:
        # Try Gemini first, fallback to OpenAI
        rag = BoardGameRAG()  # Will auto-detect provider
        rag.initialize("../data/bgg_dataset.csv")
        
        # Test query
        result = rag.query("I want a strategic game for 2-4 players that takes about 90 minutes")
        print("Response:", result['response'])
        print("Provider:", result['metadata']['provider'])
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have either GEMINI_API_KEY or OPENAI_API_KEY set.")