"""
API models for the Board Game Recommendation System.
Defines Pydantic models for request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SortOption(str, Enum):
    """Available sorting options for recommendations."""
    rating = "Rating Average"
    complexity = "Complexity Average"
    year = "Year Published"
    rank = "BGG Rank"


class RecommendationRequest(BaseModel):
    """Request model for game recommendations."""
    num_players: Optional[int] = Field(None, ge=1, le=20, description="Number of players")
    duration_min: Optional[int] = Field(None, ge=1, description="Minimum play time in minutes")
    duration_max: Optional[int] = Field(None, ge=1, description="Maximum play time in minutes")
    complexity_min: Optional[float] = Field(None, ge=1.0, le=5.0, description="Minimum complexity (1-5 scale)")
    complexity_max: Optional[float] = Field(None, ge=1.0, le=5.0, description="Maximum complexity (1-5 scale)")
    min_rating: Optional[float] = Field(None, ge=0.0, le=10.0, description="Minimum BGG rating")
    max_age: Optional[int] = Field(None, ge=1, le=99, description="Maximum recommended age")
    mechanics: Optional[List[str]] = Field(None, description="List of desired game mechanics")
    domains: Optional[List[str]] = Field(None, description="List of desired game domains")
    n_recommendations: Optional[int] = Field(10, ge=1, le=50, description="Number of recommendations to return")
    sort_by: Optional[SortOption] = Field(SortOption.rating, description="Sort recommendations by this field")

    class Config:
        json_schema_extra = {
            "example": {
                "num_players": 4,
                "duration_min": 30,
                "duration_max": 90,
                "complexity_min": 2.0,
                "complexity_max": 3.5,
                "min_rating": 7.0,
                "max_age": 12,
                "mechanics": ["Dice Rolling", "Worker Placement"],
                "domains": ["Strategy Games"],
                "n_recommendations": 5,
                "sort_by": "Rating Average"
            }
        }


class SimilarGamesRequest(BaseModel):
    """Request model for finding similar games."""
    game_name: str = Field(..., min_length=1, description="Name of the game to find similar games for")
    n_recommendations: Optional[int] = Field(10, ge=1, le=50, description="Number of similar games to return")

    class Config:
        json_schema_extra = {
            "example": {
                "game_name": "Catan",
                "n_recommendations": 5
            }
        }


class GameInfo(BaseModel):
    """Model for individual game information."""
    name: str = Field(..., description="Game name")
    year_published: Optional[int] = Field(None, description="Year the game was published")
    min_players: Optional[int] = Field(None, description="Minimum number of players")
    max_players: Optional[int] = Field(None, description="Maximum number of players")
    play_time: Optional[int] = Field(None, description="Play time in minutes")
    min_age: Optional[int] = Field(None, description="Minimum recommended age")
    rating_average: Optional[float] = Field(None, description="Average BGG rating")
    complexity_average: Optional[float] = Field(None, description="Average complexity (1-5 scale)")
    bgg_rank: Optional[int] = Field(None, description="BGG rank")
    mechanics: Optional[str] = Field(None, description="Game mechanics")
    domains: Optional[str] = Field(None, description="Game domains")
    similarity_score: Optional[float] = Field(None, description="Similarity score (for similar games)")


class RecommendationResponse(BaseModel):
    """Response model for game recommendations."""
    games: List[GameInfo] = Field(..., description="List of recommended games")
    total_found: int = Field(..., description="Total number of games found")
    filters_applied: dict = Field(..., description="Summary of applied filters")

    class Config:
        json_schema_extra = {
            "example": {
                "games": [
                    {
                        "name": "Wingspan",
                        "year_published": 2019,
                        "min_players": 1,
                        "max_players": 5,
                        "play_time": 70,
                        "min_age": 10,
                        "rating_average": 8.1,
                        "complexity_average": 2.4,
                        "bgg_rank": 15,
                        "mechanics": "Card Drafting, Dice Rolling, Hand Management",
                        "domains": "Strategy Games"
                    }
                ],
                "total_found": 1,
                "filters_applied": {
                    "num_players": 4,
                    "complexity_max": 3.0,
                    "min_rating": 7.0
                }
            }
        }


class SimilarGamesResponse(BaseModel):
    """Response model for similar games."""
    target_game: str = Field(..., description="Name of the target game")
    similar_games: List[GameInfo] = Field(..., description="List of similar games")
    total_found: int = Field(..., description="Total number of similar games found")

    class Config:
        json_schema_extra = {
            "example": {
                "target_game": "Catan",
                "similar_games": [
                    {
                        "name": "Chinatown",
                        "year_published": 1999,
                        "min_players": 3,
                        "max_players": 5,
                        "play_time": 90,
                        "min_age": 12,
                        "rating_average": 7.2,
                        "complexity_average": 2.8,
                        "similarity_score": 0.85
                    }
                ],
                "total_found": 1
            }
        }


class MechanicsResponse(BaseModel):
    """Response model for popular mechanics."""
    mechanics: List[str] = Field(..., description="List of popular game mechanics")
    total_count: int = Field(..., description="Total number of mechanics returned")

    class Config:
        json_schema_extra = {
            "example": {
                "mechanics": ["Dice Rolling", "Hand Management", "Set Collection"],
                "total_count": 3
            }
        }


class DomainsResponse(BaseModel):
    """Response model for popular domains."""
    domains: List[str] = Field(..., description="List of available game domains")
    total_count: int = Field(..., description="Total number of domains returned")

    class Config:
        json_schema_extra = {
            "example": {
                "domains": ["Strategy Games", "Family Games", "Thematic Games"],
                "total_count": 3
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="API status")
    system_ready: bool = Field(..., description="Whether the recommendation system is ready")
    games_loaded: int = Field(..., description="Number of games in the database")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "system_ready": True,
                "games_loaded": 20344
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "No games found",
                "detail": "No games match the specified criteria. Try relaxing some filters."
            }
        }