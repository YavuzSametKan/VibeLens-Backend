import json
import concurrent.futures
import google.generativeai as genai
from app.core.config import settings
from app.schemas.analysis import Category
from app.utils.timer import ExecutionTimer
from app.core.prompts import build_gemini_prompt
# NOT: app/core/prompts.py dosyasının var olduğu varsayılmıştır.

from app.services.search_service import get_content_metadata

# Gemini Config
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel(
    'gemini-flash-latest',
    generation_config={"response_mime_type": "application/json"}
)

def update_item_with_metadata(item, category):
    """
    Gemini verisi ile API verisini akıllıca birleştirir.
    """
    try:
        metadata = get_content_metadata(item['title'], item['creator'], category)

        # Poster: Her zaman API'den al
        item['poster_url'] = metadata['poster']

        # Diğer verileri API doluysa güncelle
        if metadata.get('overview'):
            item['overview'] = metadata['overview']
        if metadata.get('rating'):
            item['rating'] = metadata['rating']
        if metadata.get('year'):
            item['year'] = metadata['year']
        if metadata.get('external_links'):
            item['external_links'] = metadata['external_links']

    except Exception as e:
        print(f"Metadata merge hatası ({item.get('title')}): {e}")
    return item

def get_secondary_emotion(scores, dominant):
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for emotion, score in sorted_scores:
        if emotion == dominant or emotion == "Neutral":
            continue
        if score > 0.01:
            return emotion
    return "None"

def get_recommendations_from_gemini(user_context, category: Category):
    try:
        dominant = user_context['emotion']
        raw_scores = user_context['raw_emotion_scores']
        secondary = get_secondary_emotion(raw_scores, dominant)

        with ExecutionTimer("Prompt Engineering"):
            prompt = build_gemini_prompt(
                category=category,
                age=user_context['age'],
                gender=user_context['gender'],
                emotion=dominant,
                secondary_emotion=secondary,
                raw_scores=raw_scores
            )

        with ExecutionTimer(f"Gemini AI ({category.value})"):
            response = model.generate_content(prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)

        recommendations = data.get('recommendations', [])

        with ExecutionTimer(f"Metadata Zenginleştirme ({len(recommendations)} Adet)"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(update_item_with_metadata, item, category)
                    for item in recommendations
                ]
                concurrent.futures.wait(futures)

        return data

    except Exception as e:
        print(f"AI Servis Hatası: {e}")
        return None