from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Optional

# --- ENUMERATIONS ---

# Category Enum: Ensures a fixed set of content types and enables dropdowns in the API/Frontend
class Category(str, Enum):
    MOVIE = "Movie"
    SERIES = "Series"
    MUSIC = "Music"
    BOOK = "Book"

# --- PYDANTIC SCHEMAS ---

# Template for a single recommendation card
class RecommendationItem(BaseModel):
    title: str
    creator: Optional[str] = ""
    rating: Optional[str] = None
    poster_url: Optional[str] = None
    overview: Optional[str] = None
    year: Optional[str] = None
    reason: str
    external_links: Optional[Dict[str, str]] = None

# The Main Response Schema returned by the API
class VibeResponse(BaseModel):
    mood_title: str
    mood_description: str
    dominant_emotion: str
    secondary_emotion: str
    detected_age: int
    detected_gender: str
    emotion_scores: Dict[str, float]
    recommendations: List[RecommendationItem]