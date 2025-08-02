"""
FastAPI application for the Board Game Recommendation System.
Provides REST API endpoints with automatic Swagger documentation.
"""

import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessor import BoardGameDataPreprocessor
from recommendation_engine import BoardGameRecommender
from api_models import (
    RecommendationRequest, SimilarGamesRequest, RecommendationResponse, 
    SimilarGamesResponse, MechanicsResponse, DomainsResponse, HealthResponse,
    ErrorResponse, GameInfo, SortOption
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global recommendation system instance
recommendation_system = None

# FastAPI app configuration
app = FastAPI(
    title="Board Game Recommendation API",
    description="""
    ## AI-Powered Board Game Recommendation System
    
    This API provides intelligent board game recommendations based on your preferences.
    
    ### Features:
    - **Filter-based recommendations**: Find games based on player count, duration, complexity, and more
    - **Similarity-based recommendations**: Discover games similar to ones you already enjoy
    - **Browse game metadata**: Explore popular mechanics and domains
    - **Comprehensive game database**: Access to 20,000+ board games from BoardGameGeek
    
    ### How to use:
    1. Use `/recommendations` endpoint for personalized game suggestions
    2. Use `/similar` endpoint to find games similar to a specific title
    3. Browse `/mechanics` and `/domains` to see available filters
    4. Check `/health` to verify the system status
    
    All endpoints return detailed game information including ratings, complexity, player counts, and more.
    """,
    version="1.0.0",
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
        domains=str(game_row.get('Domains', '')) if pd.notna(game_row.get('Domains')) else None,
        similarity_score=float(game_row.get('Similarity_Score')) if pd.notna(game_row.get('Similarity_Score')) else None
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check the health and status of the recommendation system.
    
    Returns system status and database information.
    """
    try:
        system = get_recommendation_system()
        return HealthResponse(
            status="healthy",
            system_ready=system.is_ready,
            games_loaded=system.games_count
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            system_ready=False,
            games_loaded=0
        )


@app.post("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    system: RecommendationSystem = Depends(get_recommendation_system)
):
    """
    Get personalized board game recommendations based on your preferences.
    
    Filter games by various criteria including:
    - Number of players
    - Play duration
    - Complexity level
    - Minimum rating
    - Age appropriateness
    - Game mechanics and domains
    
    Returns a list of games matching your criteria, sorted by the specified field.
    """
    try:
        # Extract request parameters
        params = request.dict(exclude_none=True)
        sort_by = params.pop('sort_by', 'Rating Average')
        
        # Get recommendations
        recommendations = system.recommender.recommend_games(
            sort_by=sort_by,
            **params
        )
        
        if recommendations.empty:
            raise HTTPException(
                status_code=404, 
                detail="No games found matching your criteria. Try relaxing some filters."
            )
        
        # Convert to response format
        games = [convert_game_to_info(row) for _, row in recommendations.iterrows()]
        
        # Filter out None values for filters_applied
        filters_applied = {k: v for k, v in request.dict().items() if v is not None and k not in ['n_recommendations', 'sort_by']}
        
        return RecommendationResponse(
            games=games,
            total_found=len(games),
            filters_applied=filters_applied
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar", response_model=SimilarGamesResponse, tags=["Recommendations"])
async def get_similar_games(
    request: SimilarGamesRequest,
    system: RecommendationSystem = Depends(get_recommendation_system)
):
    """
    Find board games similar to a specific game.
    
    Uses machine learning to analyze game features and find the most similar games
    based on mechanics, themes, complexity, and other characteristics.
    
    Perfect for discovering new games if you enjoyed a particular title.
    """
    try:
        similar_games = system.recommender.get_similar_games(
            game_name=request.game_name,
            n_recommendations=request.n_recommendations
        )
        
        if similar_games.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No game found matching '{request.game_name}'. Try a different name or check spelling."
            )
        
        # Convert to response format
        games = [convert_game_to_info(row) for _, row in similar_games.iterrows()]
        
        return SimilarGamesResponse(
            target_game=request.game_name,
            similar_games=games,
            total_found=len(games)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similar games: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mechanics", response_model=MechanicsResponse, tags=["Browse"])
async def get_popular_mechanics(
    top_n: int = Query(20, ge=1, le=100, description="Number of top mechanics to return"),
    system: RecommendationSystem = Depends(get_recommendation_system)
):
    """
    Get the most popular board game mechanics.
    
    Mechanics describe how games are played (e.g., "Dice Rolling", "Worker Placement").
    Use this endpoint to see available mechanics for filtering recommendations.
    
    Returns mechanics ordered by popularity in the database.
    """
    try:
        mechanics = system.recommender.get_popular_mechanics(top_n)
        
        return MechanicsResponse(
            mechanics=mechanics,
            total_count=len(mechanics)
        )
        
    except Exception as e:
        logger.error(f"Error getting mechanics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/domains", response_model=DomainsResponse, tags=["Browse"])
async def get_popular_domains(
    top_n: int = Query(10, ge=1, le=50, description="Number of top domains to return"),
    system: RecommendationSystem = Depends(get_recommendation_system)
):
    """
    Get the most popular board game domains (categories).
    
    Domains describe broad game categories (e.g., "Strategy Games", "Family Games").
    Use this endpoint to see available domains for filtering recommendations.
    
    Returns domains ordered by popularity in the database.
    """
    try:
        domains = system.recommender.get_popular_domains(top_n)
        
        return DomainsResponse(
            domains=domains,
            total_count=len(domains)
        )
        
    except Exception as e:
        logger.error(f"Error getting domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations_get(
    num_players: Optional[int] = Query(None, ge=1, le=20, description="Number of players"),
    duration_min: Optional[int] = Query(None, ge=1, description="Minimum play time in minutes"),
    duration_max: Optional[int] = Query(None, ge=1, description="Maximum play time in minutes"),
    complexity_min: Optional[float] = Query(None, ge=1.0, le=5.0, description="Minimum complexity (1-5)"),
    complexity_max: Optional[float] = Query(None, ge=1.0, le=5.0, description="Maximum complexity (1-5)"),
    min_rating: Optional[float] = Query(None, ge=0.0, le=10.0, description="Minimum BGG rating"),
    max_age: Optional[int] = Query(None, ge=1, le=99, description="Maximum recommended age"),
    n_recommendations: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    sort_by: SortOption = Query(SortOption.rating, description="Sort recommendations by"),
    system: RecommendationSystem = Depends(get_recommendation_system)
):
    """
    Get board game recommendations using query parameters (alternative to POST).
    
    This is a convenience endpoint that accepts parameters via URL query string
    instead of a JSON request body. Useful for simple requests and testing.
    
    For more complex filtering (including mechanics and domains), use the POST endpoint.
    """
    try:
        # Create request object
        request = RecommendationRequest(
            num_players=num_players,
            duration_min=duration_min,
            duration_max=duration_max,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
            min_rating=min_rating,
            max_age=max_age,
            n_recommendations=n_recommendations,
            sort_by=sort_by
        )
        
        # Use the POST endpoint logic
        return await get_recommendations(request, system)
        
    except Exception as e:
        logger.error(f"Error in GET recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar", response_model=SimilarGamesResponse, tags=["Recommendations"])
async def get_similar_games_get(
    game_name: str = Query(..., description="Name of the game to find similar games for"),
    n_recommendations: int = Query(10, ge=1, le=50, description="Number of similar games"),
    system: RecommendationSystem = Depends(get_recommendation_system)
):
    """
    Find similar games using query parameters (alternative to POST).
    
    This is a convenience endpoint that accepts parameters via URL query string
    instead of a JSON request body.
    """
    try:
        request = SimilarGamesRequest(
            game_name=game_name,
            n_recommendations=n_recommendations
        )
        
        return await get_similar_games(request, system)
        
    except Exception as e:
        logger.error(f"Error in GET similar games: {e}")
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