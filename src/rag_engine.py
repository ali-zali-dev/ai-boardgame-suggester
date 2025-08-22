"""
LlamaIndex-based RAG (Retrieval-Augmented Generation) engine for board game recommendations.
Lightweight replacement for ChromaDB + sentence-transformers setup.
"""

import pandas as pd
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini

# Load environment variables
load_dotenv()


class BoardGameRAG:
    """Lightweight RAG system for board game recommendations using LlamaIndex."""
    
    def __init__(self, provider_name: Optional[str] = None):
        """
        Initialize the RAG system with LlamaIndex.
        
        Args:
            provider_name: LLM provider to use ("openai", "gemini", or None for auto-detect)
        """
        # Initialize LLM provider
        self.llm = self._initialize_llm(provider_name)
        
        # Initialize lightweight embedding model (replaces heavy sentence-transformers)
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",  # Much smaller than sentence-transformers
            cache_folder="./embeddings_cache"
        )
        
        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        self.index = None
        self.query_engine = None
        self.games_df = None
        
        print(f"✅ LlamaIndex RAG initialized with {type(self.llm).__name__}")
    
    def _initialize_llm(self, provider_name: Optional[str] = None) -> Any:
        """Initialize LLM provider with auto-detection."""
        # Check environment variables
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        
        if not has_openai and not has_gemini:
            raise ValueError(
                "No LLM API key found! Please set either:\n"
                "- OPENAI_API_KEY for OpenAI\n"
                "- GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini"
            )
        
        # Use specified provider or auto-detect
        preferred_provider = provider_name or os.getenv("LLM_PROVIDER", "gemini").lower()
        
        if preferred_provider == "gemini" and has_gemini:
            return Gemini(
                model="models/gemini-1.5-flash",
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            )
        elif preferred_provider == "openai" and has_openai:
            return OpenAI(model="gpt-3.5-turbo", temperature=0.7)
        elif has_gemini:
            return Gemini(
                model="models/gemini-1.5-flash",
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            )
        elif has_openai:
            return OpenAI(model="gpt-3.5-turbo", temperature=0.7)
        else:
            raise ValueError("No compatible LLM provider available")
    
    def load_and_process_data(self, csv_path: str) -> None:
        """Load board game data and create LlamaIndex documents."""
        print("Loading board game data...")
        
        # Load CSV
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        print(f"Loaded {len(df)} games")
        
        # Clean data
        df = self._clean_data(df)
        self.games_df = df
        
        # Convert to LlamaIndex documents
        documents = []
        for idx, row in df.iterrows():
            # Create rich game description
            description = self._create_game_description(row)
            
            # Create LlamaIndex document with metadata
            doc = Document(
                text=description,
                metadata={
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
            )
            documents.append(doc)
        
        print(f"Created {len(documents)} documents")
        
        # Create vector index (replaces ChromaDB)
        print("Creating vector index...")
        self.index = VectorStoreIndex.from_documents(documents)
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )
        
        print("✅ RAG system ready!")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the board game data."""
        # Replace commas with dots in numeric columns
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
        """Create a rich text description for each game."""
        mechanics = row.get('Mechanics', '').replace(',', ', ')
        domains = row.get('Domains', '').replace(',', ', ')
        name = row.get('Name', 'Unknown')
        
        description = f"""
        Game: {name}
        Type: {domains}
        Mechanics: {mechanics}
        Players: {row.get('Min Players', 1)}-{row.get('Max Players', 8)}
        Duration: {row.get('Play Time', 60)} minutes
        Age: {row.get('Min Age', 8)}+ years
        Complexity: {row.get('Complexity Average', 0):.1f}/5
        Rating: {row.get('Rating Average', 0):.1f}/10
        BGG Rank: #{row.get('BGG Rank', 'Unranked')}
        
        This {domains.lower()} game features {mechanics.lower()} mechanics.
        Suitable for {row.get('Min Players', 1)} to {row.get('Max Players', 8)} players.
        Playing time is approximately {row.get('Play Time', 60)} minutes.
        Complexity rating: {row.get('Complexity Average', 0):.1f} ({'beginner' if row.get('Complexity Average', 0) < 2.5 else 'intermediate' if row.get('Complexity Average', 0) < 3.5 else 'advanced'} level)
        Community rating: {row.get('Rating Average', 0):.1f} ({'excellent' if row.get('Rating Average', 0) >= 8.0 else 'very good' if row.get('Rating Average', 0) >= 7.0 else 'good' if row.get('Rating Average', 0) >= 6.0 else 'average'} quality)
        """
        
        return description.strip()
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Query the RAG system for board game recommendations."""
        if self.query_engine is None:
            raise ValueError("RAG system not initialized. Call load_and_process_data() first.")
        
        # Enhanced prompt for better responses
        enhanced_query = f"""
        You are a knowledgeable board game expert. Based on the following request:
        "{user_query}"
        
        Provide specific game recommendations with:
        1. Game names and brief descriptions
        2. Why each game fits the user's criteria
        3. Player count, duration, and complexity details
        4. Any additional insights about gameplay
        
        Be enthusiastic and helpful in your response.
        """
        
        # Query using LlamaIndex
        response = self.query_engine.query(enhanced_query)
        
        # Extract source nodes (similar games)
        source_games = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                if hasattr(node, 'metadata'):
                    source_games.append({
                        'name': node.metadata.get('name', 'Unknown'),
                        'rating': node.metadata.get('rating', 0),
                        'complexity': node.metadata.get('complexity', 0),
                        'players': f"{node.metadata.get('min_players', 1)}-{node.metadata.get('max_players', 8)}",
                        'time': f"{node.metadata.get('play_time', 60)} min",
                        'mechanics': node.metadata.get('mechanics', ''),
                        'score': getattr(node, 'score', 0)
                    })
        
        return {
            'query': user_query,
            'response': str(response),
            'games': source_games[:5],  # Top 5 recommendations
            'metadata': {
                'num_results': len(source_games),
                'provider': type(self.llm).__name__,
                'embedding_model': 'BAAI/bge-small-en-v1.5'
            }
        }
    
    def initialize(self, csv_path: str) -> None:
        """Initialize the entire RAG system."""
        print("Initializing LlamaIndex Board Game RAG System...")
        self.load_and_process_data(csv_path)
        print("🚀 RAG system ready!")


if __name__ == "__main__":
    # Example usage
    try:
        rag = BoardGameRAG()  # Auto-detect provider
        rag.initialize("../data/bgg_dataset.csv")
        
        # Test query
        result = rag.query("I want a strategic game for 2-4 players that takes about 90 minutes")
        print("Response:", result['response'])
        print("Provider:", result['metadata']['provider'])
        print("Games found:", len(result['games']))
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have either OPENAI_API_KEY or GEMINI_API_KEY set.")