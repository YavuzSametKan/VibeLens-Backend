from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Optional

# Kategori Enum'ı: API'de Dropdown açılmasını sağlar
class Category(str, Enum):
    MOVIE = "Movie"
    SERIES = "Series"
    MUSIC = "Music"
    BOOK = "Book"

# Tavsiye Kartının Şablonu
class RecommendationItem(BaseModel):
    title: str
    creator: Optional[str] = ""
    rating: Optional[str] = None
    poster_url: Optional[str] = None
    overview: Optional[str] = None
    year: Optional[str] = None
    reason: str
    external_links: Optional[Dict[str, str]] = None

# API'nin Döneceği Ana Cevap Şablonu
class VibeResponse(BaseModel):
    mood_title: str
    mood_description: str
    dominant_emotion: str
    secondary_emotion: str
    detected_age: int
    detected_gender: str
    emotion_scores: Dict[str, float]
    recommendations: List[RecommendationItem]