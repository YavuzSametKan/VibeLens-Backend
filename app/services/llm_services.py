import json
import concurrent.futures
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from app.core.config import settings
from app.schemas.analysis import Category
from app.utils.timer import ExecutionTimer
from app.core.prompts import build_gemini_prompt

from app.services.search_service import get_content_metadata

# --- CONFIGURATION ---
MAX_RETRIES = 3  # Maximum number of retry attempts
RETRY_DELAY = 2  # Delay in seconds between retries

# --- GEMINI API CLIENT SETUP ---
genai.configure(api_key=settings.GEMINI_API_KEY)

safety_settings = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

model = genai.GenerativeModel(
    model_name='gemini-flash-latest',
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.9,  # Increased for higher creativity
        "top_p": 0.95,  # Broadened word selection pool
    },
    safety_settings=safety_settings
)


# --- HELPER FUNCTIONS ---
def update_item_with_metadata(item: dict, category: Category) -> dict:
    """
    Fetches metadata for a content item and merges it with the Gemini recommendation.
    """
    try:
        title = item.get('title', '')
        metadata = get_content_metadata(title, category.value)
        
        # Merge metadata with item, only if metadata has valid values
        if metadata.get('rating') and str(metadata['rating']).strip():
            item['rating'] = metadata['rating']
        
        if metadata.get('overview') and str(metadata['overview']).strip():
            item['overview'] = metadata['overview']
        
        if metadata.get('year') and str(metadata['year']).strip():
            item['year'] = metadata['year']
        
        if metadata.get('external_links'):
            item['external_links'] = metadata['external_links']

    except Exception as e:
        print(f"Error merging metadata for {item.get('title')}: {e}")
    return item


def get_fallback_response() -> dict:
    """
    Returns a default response if the AI service completely fails.
    """
    return {
        "mood_title": "Connection Issue",
        "mood_description": "The AI service is currently busy, but I can fetch random popular content for you.",
        "recommendations": []  # Empty list indicates 'No Recommendations'
    }


# --- MAIN LOGIC ---
def get_recommendations_from_gemini(user_context: dict, category: Category) -> dict:
    # 1. Prompt Preparation
    try:
        dominant = user_context['emotion']
        raw_scores = user_context['raw_emotion_scores']
        secondary = user_context['secondary_emotion']

        prompt = build_gemini_prompt(
            category=category,
            age=user_context['age'],
            gender=user_context['gender'],
            emotion=dominant,
            secondary_emotion=secondary,
            raw_scores=raw_scores
        )
    except Exception as e:
        print(f"Prompt Building Error: {e}")
        return get_fallback_response()

    # 2. Retry Mechanism
    data = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with ExecutionTimer(f"Gemini AI ({category.value}) - Attempt {attempt}/{MAX_RETRIES}"):
                response = model.generate_content(prompt)

                # Check for empty response (e.g., due to safety block)
                try:
                    raw_text = response.text
                except ValueError:
                    print(
                        f" Attempt {attempt}: Gemini returned an empty response. Reason: {response.prompt_feedback}")
                    raise ValueError("Empty Response from Gemini")

                # Clean and parse JSON
                clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)

                # Success, break the loop
                break

        except Exception as e:
            print(f" Attempt {attempt} Failed: {e}")
            if attempt < MAX_RETRIES:
                print(f" Waiting for {RETRY_DELAY} seconds before retry...")
                time.sleep(RETRY_DELAY)
            else:
                print(" All attempts failed.")

    # 3. Fallback Check
    if not data:
        print(" Returning emergency fallback data.")
        return get_fallback_response()

    # 4. Metadata Enrichment (Run only if data was successfully fetched)
    try:
        recommendations = data.get('recommendations', [])

        if recommendations:
            with ExecutionTimer(f"Metadata Enrichment ({len(recommendations)} Items)"):
                # Use ThreadPoolExecutor for concurrent fetching to speed up I/O-bound tasks
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(update_item_with_metadata, item, category)
                        for item in recommendations
                    ]
                    # Wait for all futures to complete
                    concurrent.futures.wait(futures)

        return data

    except Exception as e:
        print(f"Metadata Processing Error: {e}")
        # Return the raw Gemini data even if metadata merging failed
        return data