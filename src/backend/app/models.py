from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class MatchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="Person description to match against")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of matches to return")
    min_score: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum normalized score threshold (0-1)")

class MatchResult(BaseModel):
    name: str
    employment: str
    board_service: str
    score: float
    rank: int

class MatchResponse(BaseModel):
    query: str
    total_matches: int
    matches: List[MatchResult]

class UploadResponse(BaseModel):
    status: str
    message: str
    rows_loaded: int
    columns: List[str]

class HealthResponse(BaseModel):
    status: str
    dataset_loaded: bool
    dataset_size: int = 0

class ExportRequest(BaseModel):
    matches: List[MatchResult]