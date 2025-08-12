"""
FastAPI application for the Board Game Recommendation System.
Provides simplified REST API with only 2 endpoints.
"""

import os
import sys
import pandas as pd
import json
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessor import BoardGameDataPreprocessor
from recommendation_engine import BoardGameRecommender
from api_models import RecommendationResponse, GameInfo, SortOption

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini client
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key or gemini_api_key == 'your_gemini_api_key_here':
    logger.warning("GEMINI_API_KEY not set. Natural language queries will not work.")
    gemini_client = None
else:
    try:
        # Create Gemini client using the new SDK
        gemini_client = genai.Client(api_key=gemini_api_key)
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        gemini_client = None

# Global recommendation system instance
recommendation_system = None

# FastAPI app configuration
app = FastAPI(
    title="Board Game Recommendation API",
    description="""
    ## AI-Powered Board Game Recommendation API
    
    This API provides intelligent board game recommendations with just 2 endpoints:
    
    ### Endpoints:
    1. **GET /recommendations** - Filter-based recommendations using query parameters
    2. **GET /query** - AI-powered natural language query recommendations
    
    ### Features:
    - **20,000+ board games** from BoardGameGeek database
    - **Smart filtering** by player count, duration, complexity, and rating
    - **Google Gemini AI** for advanced natural language understanding
    - **Fallback parsing** when AI is unavailable
    - **Comprehensive game data** including ratings, mechanics, and more
    
    ### AI Query Examples:
    - "I want a strategic but easy game for 3 players"
    - "Find me a quick party game for 6 people under 30 minutes"
    - "Suggest complex strategy games for 2 players with high ratings"
    - "Family-friendly games with dice rolling for 4 players"
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendationSystem:
    """Singleton wrapper for the recommendation system."""
    
    def __init__(self):
        self.recommender = None
        self.is_ready = False
        self.games_count = 0
        
    def initialize(self):
        """Initialize the recommendation system."""
        try:
            logger.info("Initializing Board Game Recommendation System...")
            
            # Paths to data files
            processed_data_path = 'data/processed_games.csv'
            original_data_path = 'data/bgg_dataset.csv'
            
            # Check if data files exist
            if not os.path.exists(original_data_path):
                raise FileNotFoundError(f"Dataset not found: {original_data_path}")
            
            # Preprocess data if needed
            if not os.path.exists(processed_data_path):
                logger.info("Processing dataset for first time...")
                preprocessor = BoardGameDataPreprocessor()
                df = preprocessor.load_data(original_data_path)
                processed_df = preprocessor.preprocess_features(df)
                
                # Ensure data directory exists
                os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
                processed_df.to_csv(processed_data_path, index=False)
                
                # Save preprocessor
                os.makedirs('models', exist_ok=True)
                preprocessor.save_preprocessor('models/preprocessor.pkl')
                logger.info("Data preprocessing completed")
            
            # Initialize recommender
            self.recommender = BoardGameRecommender()
            self.recommender.load_data(processed_data_path, original_data_path)
            self.recommender.train_similarity_model()
            
            self.games_count = len(self.recommender.games_data)
            self.is_ready = True
            
            logger.info(f"System initialized successfully with {self.games_count} games")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            self.is_ready = False
            raise


def get_recommendation_system() -> RecommendationSystem:
    """Dependency to get the recommendation system instance."""
    global recommendation_system
    if recommendation_system is None:
        recommendation_system = RecommendationSystem()
        recommendation_system.initialize()
    
    if not recommendation_system.is_ready:
        raise HTTPException(status_code=503, detail="Recommendation system not ready")
    
    return recommendation_system


def convert_game_to_info(game_row) -> GameInfo:
    """Convert a pandas game row to GameInfo model."""
    return GameInfo(
        name=game_row.get('Name', 'Unknown'),
        year_published=int(game_row.get('Year Published')) if pd.notna(game_row.get('Year Published')) else None,
        min_players=int(game_row.get('Min Players')) if pd.notna(game_row.get('Min Players')) else None,
        max_players=int(game_row.get('Max Players')) if pd.notna(game_row.get('Max Players')) else None,
        play_time=int(game_row.get('Play Time')) if pd.notna(game_row.get('Play Time')) else None,
        min_age=int(game_row.get('Min Age')) if pd.notna(game_row.get('Min Age')) else None,
        rating_average=float(game_row.get('Rating Average')) if pd.notna(game_row.get('Rating Average')) else None,
        complexity_average=float(game_row.get('Complexity Average')) if pd.notna(game_row.get('Complexity Average')) else None,
        bgg_rank=int(game_row.get('BGG Rank')) if pd.notna(game_row.get('BGG Rank')) else None,
        mechanics=str(game_row.get('Mechanics', '')) if pd.notna(game_row.get('Mechanics')) else None,
        domains=str(game_row.get('Domains', '')) if pd.notna(game_row.get('Domains')) else None
    )


def parse_natural_language_query_with_gemini(query: str) -> tuple[dict, str]:
    """Parse natural language query using Google Gemini."""
    if not gemini_client:
        # Fallback to simple parsing if Gemini is not available
        return parse_natural_language_query_fallback(query)
    
    try:
        model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        temperature = float(os.getenv('GEMINI_TEMPERATURE', '0.3'))
        
        prompt = f"""
You are a board game recommendation assistant. Parse the following user query and extract filter parameters for board game recommendations.

User query: "{query}"

Please respond with ONLY a valid JSON object containing the extracted filters. Use these exact field names and value types:

{{
  "num_players": integer or null,
  "duration_min": integer (minutes) or null,
  "duration_max": integer (minutes) or null, 
  "complexity_min": float (1.0-5.0) or null,
  "complexity_max": float (1.0-5.0) or null,
  "min_rating": float (0.0-10.0) or null,
  "max_age": integer or null,
  "year_min": integer or null,
  "year_max": integer or null,
  "mechanics": array of strings or null,
  "domains": array of strings or null,
  "n_recommendations": integer (default 10),
  "sort_by": "Rating Average" (default),
  "interpretation": "human-readable description of what was understood"
}}

Guidelines:
- For complexity: easy/simple/light = max 2.5, medium = 2.0-3.5, complex/heavy = min 3.5
- For duration: quick/fast/short = max 45min, long = min 90min
- For rating: good/popular = min 7.0, best/top/excellent = min 8.0
- For "best" queries: use sort_by "BGG Rank" (rank 1 is best game) AND min_rating 8.0
- For "newest/new/recent" queries: use sort_by "Year Published" AND set minimum year (e.g., 2020+)
- For "old/classic/vintage" queries: use sort_by "Year Published" AND set maximum year (e.g., before 2010)
- Common domains: "Strategy Games", "Family Games", "Party Games", "Thematic Games"
- Common mechanics: "Dice Rolling", "Hand Management", "Worker Placement", "Card Drafting", "Cooperative Game"
- Set num_players to the exact number mentioned (e.g., "3 players" = 3)
- For age, extract maximum recommended age if mentioned

Only respond with the JSON object, no other text.
"""

        # Use the new SDK to generate content
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=500,
                response_modalities=['TEXT']
            )
        )
        
        # Parse the JSON response
        response_text = response.text.strip()
        
        # Remove any markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
            
        parsed_response = json.loads(response_text.strip())
        
        # Extract interpretation and remove it from filters
        interpretation = parsed_response.pop('interpretation', f"Processed query: {query}")
        
        # Clean up the filters (remove null values)
        filters = {k: v for k, v in parsed_response.items() if v is not None}
        
        logger.info(f"Gemini parsed query '{query}' -> filters: {filters}")
        return filters, interpretation
        
    except Exception as e:
        logger.warning(f"Gemini parsing failed for query '{query}': {e}. Using fallback.")
        return parse_natural_language_query_fallback(query)


def parse_natural_language_query_fallback(query: str) -> tuple[dict, str]:
    """Fallback parsing when Gemini is not available."""
    filters = {
        'n_recommendations': 10,
        'sort_by': 'Rating Average'
    }
    
    query_lower = query.lower()
    interpretation_parts = []
    
    # Simple keyword extraction
    if any(word in query_lower for word in ['strategy', 'strategic']):
        filters['domains'] = ['Strategy Games']
        interpretation_parts.append("strategy games")
        
    if any(word in query_lower for word in ['family', 'kids']):
        filters['domains'] = ['Family Games'] 
        interpretation_parts.append("family games")
        
    if any(word in query_lower for word in ['easy', 'simple', 'light']):
        filters['complexity_max'] = 2.5
        interpretation_parts.append("easy complexity")
        
    if any(word in query_lower for word in ['complex', 'heavy', 'difficult']):
        filters['complexity_min'] = 3.5
        interpretation_parts.append("complex games")
        
    if any(word in query_lower for word in ['quick', 'fast', 'short']):
        filters['duration_max'] = 45
        interpretation_parts.append("quick games")
    
    # Handle "best" queries by using BGG rank sorting and high rating filter
    if any(word in query_lower for word in ['best', 'top', 'highest', 'ranked']):
        filters['sort_by'] = 'BGG Rank'
        filters['min_rating'] = 8.0
        interpretation_parts.append("best games")
    
    # Handle "newest" queries by using year sorting and recent year filter
    if any(word in query_lower for word in ['newest', 'new', 'recent', 'latest', 'modern']):
        filters['sort_by'] = 'Year Published'
        filters['year_min'] = 2020
        interpretation_parts.append("newest games")
    
    # Handle "old" queries by using year sorting and older year filter
    if any(word in query_lower for word in ['old', 'classic', 'vintage', 'retro', 'older']):
        filters['sort_by'] = 'Year Published'
        filters['year_max'] = 2010
        interpretation_parts.append("classic games")
        
    # Extract player count
    import re
    player_match = re.search(r'(\d+)\s*(?:players?|people)', query_lower)
    if player_match:
        filters['num_players'] = int(player_match.group(1))
        interpretation_parts.append(f"{player_match.group(1)} players")
    
    interpretation = f"Fallback parsing: {', '.join(interpretation_parts)}" if interpretation_parts else "General recommendations"
    
    return filters, interpretation


@app.get("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    num_players: Optional[int] = Query(None, ge=1, le=20, description="Number of players"),
    duration_min: Optional[int] = Query(None, ge=1, description="Minimum play time in minutes"),
    duration_max: Optional[int] = Query(None, ge=1, description="Maximum play time in minutes"),
    complexity_min: Optional[float] = Query(None, ge=1.0, le=5.0, description="Minimum complexity (1-5)"),
    complexity_max: Optional[float] = Query(None, ge=1.0, le=5.0, description="Maximum complexity (1-5)"),
    min_rating: Optional[float] = Query(None, ge=0.0, le=10.0, description="Minimum BGG rating"),
    max_age: Optional[int] = Query(None, ge=1, le=99, description="Maximum recommended age"),
    year_min: Optional[int] = Query(None, ge=1900, le=2030, description="Minimum publication year"),
    year_max: Optional[int] = Query(None, ge=1900, le=2030, description="Maximum publication year"),
    n_recommendations: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    sort_by: SortOption = Query(SortOption.rating, description="Sort recommendations by"),
    system: RecommendationSystem = Depends(get_recommendation_system)
):
    """
    Get board game recommendations using specific filter parameters.
    
    Filter games by player count, duration, complexity, rating, age, and publication year.
    Perfect for when you know exactly what criteria you want.
    
    Examples: 
    - `/recommendations?num_players=4&duration_max=90&complexity_max=2.5&min_rating=7.0`
    - `/recommendations?year_min=2020&sort_by=year` (newest games)
    - `/recommendations?sort_by=rank&min_rating=8.0` (best games)
    """
    try:
        # Prepare filters
        filters = {}
        if num_players is not None:
            filters['num_players'] = num_players
        if duration_min is not None:
            filters['duration_min'] = duration_min
        if duration_max is not None:
            filters['duration_max'] = duration_max
        if complexity_min is not None:
            filters['complexity_min'] = complexity_min
        if complexity_max is not None:
            filters['complexity_max'] = complexity_max
        if min_rating is not None:
            filters['min_rating'] = min_rating
        if max_age is not None:
            filters['max_age'] = max_age
        if year_min is not None:
            filters['year_min'] = year_min
        if year_max is not None:
            filters['year_max'] = year_max
        
        # Get recommendations
        recommendations = system.recommender.recommend_games(
            n_recommendations=n_recommendations,
            sort_by=sort_by,
            **filters
        )
        
        if recommendations.empty:
            raise HTTPException(
                status_code=404, 
                detail="No games found matching your criteria. Try relaxing some filters."
            )
        
        # Convert to response format
        games = [convert_game_to_info(row) for _, row in recommendations.iterrows()]
        
        return RecommendationResponse(
            games=games,
            total_found=len(games)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations_by_query(
    q: str = Query(..., description="Natural language query (e.g., 'strategic easy game for 3 players')"),
    system: RecommendationSystem = Depends(get_recommendation_system)
):
    """
    Get board game recommendations using natural language queries.
    
    Describe what you want in plain English and get relevant recommendations.
    The system will interpret your query and convert it to appropriate filters.
    
    ### Example queries:
    - "strategic easy game for 3 players"
    - "quick party game for 6 people"
    - "complex strategy game under 90 minutes"
    - "family friendly game for 4 players"
    - "cooperative game with dice rolling"
    - "card game for 2 players, not too difficult"
    - "best strategy games for 2 players"
    - "newest family games"
    - "classic games for 4 players"
    
    ### Supported concepts:
    - **Player count**: "3 players", "for 4", "6 people"
    - **Complexity**: "easy", "simple", "complex", "heavy", "medium"
    - **Duration**: "quick", "30 minutes", "60-90 min", "long"
    - **Quality**: "good", "highly rated", "best", "top"
    - **Age**: "newest", "new", "recent", "old", "classic", "vintage"
    - **Types**: "strategy", "family", "party", "card", "dice", "cooperative"
    """
    try:
        # Parse natural language query using Gemini
        filters, interpretation = parse_natural_language_query_with_gemini(q)
        
        # Get recommendations using parsed filters
        recommendations = system.recommender.recommend_games(**filters)
        
        if recommendations.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No games found matching your query: '{q}'. Try a different description or be less specific."
            )
        
        # Convert to response format
        games = [convert_game_to_info(row) for _, row in recommendations.iterrows()]
        
        return RecommendationResponse(
            games=games,
            total_found=len(games),
            query_interpretation=interpretation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query '{q}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "detail": str(exc.detail) if hasattr(exc, 'detail') else "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)