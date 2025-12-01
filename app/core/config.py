import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

settings = Settings()