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


class RecommendationResponse(BaseModel):
    """Response model for game recommendations."""
    games: List[GameInfo] = Field(..., description="List of recommended games")
    total_found: int = Field(..., description="Total number of games found")
    query_interpretation: Optional[str] = Field(None, description="How the query was interpreted (for natural language queries)")

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
                "query_interpretation": "Looking for strategic, easy games for 3 players"
            }
        }