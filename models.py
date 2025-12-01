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
    creator: Optional[str] = "" # Yönetmen, Yazar vs.
    rating: str
    poster_url: str
    reason: str

# API'nin Döneceği Ana Cevap Şablonu
class VibeResponse(BaseModel):
    mood_title: str
    mood_description: str
    dominant_emotion: str
    detected_age: int
    detected_gender: str
    emotion_scores: Dict[str, float]
    recommendations: List[RecommendationItem]