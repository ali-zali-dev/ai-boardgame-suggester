"""
Pydantic models for the RAG-based board game recommendation API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class GameMetadata(BaseModel):
    """Metadata for a board game from the search results."""
    game_id: str
    name: str
    year: int
    min_players: int
    max_players: int
    play_time: int
    min_age: int
    rating: float
    complexity: float
    rank: int
    mechanics: str
    domains: str


class SearchResult(BaseModel):
    """A single game search result."""
    id: str
    score: float = Field(description="Similarity score (lower is better)")
    metadata: GameMetadata
    document: str = Field(description="Full game description used for embedding")


class QueryRequest(BaseModel):
    """Request model for game queries."""
    query: str = Field(description="Natural language query about board games")
    num_results: int = Field(default=5, ge=1, le=20, description="Number of games to return")


class QueryResponse(BaseModel):
    """Response model for game queries."""
    query: str
    response: str = Field(description="Generated natural language response")
    games: List[SearchResult] = Field(description="Relevant games found")
    metadata: Dict[str, Any] = Field(description="Query metadata")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    rag_initialized: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None