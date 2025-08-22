"""
FastAPI application for the RAG-based board game recommendation system.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from .rag_engine import BoardGameRAG
from .rag_models import (
    QueryRequest, QueryResponse, HealthResponse, ErrorResponse,
    SearchResult, GameMetadata
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG instance
rag_system: BoardGameRAG = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global rag_system
    
    try:
        logger.info("Initializing RAG system...")
        
        # Get preferred provider from environment
        provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()
        logger.info(f"Using LLM provider: {provider_name}")
        
        rag_system = BoardGameRAG(provider_name=provider_name)
        
        # Initialize with the dataset
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bgg_dataset.csv")
        await asyncio.get_event_loop().run_in_executor(None, rag_system.initialize, csv_path)
        
        logger.info("RAG system initialized successfully!")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        logger.error("Make sure you have either OPENAI_API_KEY or GEMINI_API_KEY set.")
        raise
    finally:
        logger.info("Shutting down RAG system...")


# Create FastAPI app
app = FastAPI(
    title="Board Game RAG API",
    description="Retrieval-Augmented Generation API for board game recommendations",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_system() -> BoardGameRAG:
    """Dependency to get the RAG system instance."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_system


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Board Game RAG API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        rag_initialized = rag_system is not None and rag_system.query_engine is not None
        return HealthResponse(
            status="healthy" if rag_initialized else "initializing",
            message="RAG system is ready" if rag_initialized else "RAG system is initializing",
            rag_initialized=rag_initialized
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Error: {str(e)}",
            rag_initialized=False
        )


@app.post("/query", response_model=QueryResponse)
async def query_games(
    request: QueryRequest,
    rag: BoardGameRAG = Depends(get_rag_system)
):
    """
    Query board games using natural language.
    
    Examples:
    - "I want a strategic game for 2-4 players"
    - "Show me cooperative games that take less than 60 minutes"
    - "What are some good gateway games for beginners?"
    - "I need a party game for 6+ players"
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Run the query in a thread executor to avoid blocking
        result = await asyncio.get_event_loop().run_in_executor(
            None, rag.query, request.query
        )
        
        # Convert search results to response model
        search_results = []
        for game_result in result['games']:
            # LlamaIndex returns simplified game data
            metadata = GameMetadata(
                game_id=str(game_result.get('name', 'unknown')),
                name=game_result.get('name', 'Unknown'),
                year=game_result.get('year', 0) if isinstance(game_result.get('year'), int) else 0,
                min_players=1,
                max_players=8,
                play_time=60,
                min_age=8,
                rating=game_result.get('rating', 0),
                complexity=game_result.get('complexity', 0),
                rank=999999,
                mechanics=game_result.get('mechanics', ''),
                domains='Strategy'
            )
            search_result = SearchResult(
                id=f"game_{game_result.get('name', 'unknown')}",
                score=game_result.get('score', 0),
                metadata=metadata,
                document=f"Game: {game_result.get('name', 'Unknown')}"
            )
            search_results.append(search_result)
        
        response = QueryResponse(
            query=result['query'],
            response=result['response'],
            games=search_results,
            metadata=result['metadata']
        )
        
        logger.info(f"Query processed successfully, returned {len(search_results)} games")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/search", response_model=List[SearchResult])
async def search_games(
    q: str,
    limit: int = 5,
    rag: BoardGameRAG = Depends(get_rag_system)
):
    """
    Search for games without LLM response generation.
    Returns raw similarity search results.
    """
    try:
        if limit < 1 or limit > 20:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 20")
        
        logger.info(f"Searching games: {q}")
        
        # Run search in thread executor  
        result = await asyncio.get_event_loop().run_in_executor(
            None, rag.query, q
        )
        
        # Convert to response model
        search_results = []
        for game_result in result['games']:
            metadata = GameMetadata(
                game_id=str(game_result.get('name', 'unknown')),
                name=game_result.get('name', 'Unknown'),
                year=game_result.get('year', 0) if isinstance(game_result.get('year'), int) else 0,
                min_players=1,
                max_players=8,
                play_time=60,
                min_age=8,
                rating=game_result.get('rating', 0),
                complexity=game_result.get('complexity', 0),
                rank=999999,
                mechanics=game_result.get('mechanics', ''),
                domains='Strategy'
            )
            search_result = SearchResult(
                id=f"game_{game_result.get('name', 'unknown')}",
                score=game_result.get('score', 0),
                metadata=metadata,
                document=f"Game: {game_result.get('name', 'Unknown')}"
            )
            search_results.append(search_result)
        
        logger.info(f"Search completed, found {len(search_results)} games")
        return search_results
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching games: {str(e)}"
        )


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats(rag: BoardGameRAG = Depends(get_rag_system)):
    """Get statistics about the game database."""
    try:
        if rag.games_df is None:
            raise HTTPException(status_code=503, detail="Game data not loaded")
        
        df = rag.games_df
        
        stats = {
            "total_games": len(df),
            "year_range": {
                "min": int(df['Year Published'].min()) if 'Year Published' in df.columns else None,
                "max": int(df['Year Published'].max()) if 'Year Published' in df.columns else None
            },
            "player_count_range": {
                "min": int(df['Min Players'].min()) if 'Min Players' in df.columns else None,
                "max": int(df['Max Players'].max()) if 'Max Players' in df.columns else None
            },
            "rating_range": {
                "min": float(df['Rating Average'].min()) if 'Rating Average' in df.columns else None,
                "max": float(df['Rating Average'].max()) if 'Rating Average' in df.columns else None,
                "mean": float(df['Rating Average'].mean()) if 'Rating Average' in df.columns else None
            },
            "complexity_range": {
                "min": float(df['Complexity Average'].min()) if 'Complexity Average' in df.columns else None,
                "max": float(df['Complexity Average'].max()) if 'Complexity Average' in df.columns else None,
                "mean": float(df['Complexity Average'].mean()) if 'Complexity Average' in df.columns else None
            },
            "vector_db_size": len(rag.games_df) if rag.games_df is not None else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting database stats: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=str(exc.detail)).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)