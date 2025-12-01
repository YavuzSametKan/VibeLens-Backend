import cv2
import numpy as np
import json
import os
import concurrent.futures
from deepface import DeepFace
import google.generativeai as genai
from models import Category
from prompts import build_gemini_prompt
from search_service import get_poster_url
from dotenv import load_dotenv
from utils import ExecutionTimer  # <-- YENİ IMPORT

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    'gemini-flash-latest',
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.7,
        "max_output_tokens": 8192 # Kesilme olmasın diye limiti açtık
    },
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
)


def analyze_image_with_deepface(image_bytes):
    # Tüm işlemi Timer içine alıyoruz
    with ExecutionTimer("DeepFace Görüntü Analizi"):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # DeepFace'in asıl vakit harcadığı yer
            analysis = DeepFace.analyze(img_path=img, actions=['emotion', 'age', 'gender'], enforce_detection=False)
            result = analysis[0] if isinstance(analysis, list) else analysis

            return {
                "emotion": result['dominant_emotion'],
                "age": int(result['age']),
                "gender": result['dominant_gender'],
                "raw_emotion_scores": {k: float(v) for k, v in result['emotion'].items()}
            }
        except Exception as e:
            print(f"DeepFace Hatası: {e}")
            return None


def update_item_with_poster(item, category):
    # Tek bir posterin bulunma süresini ölçmek istersen (Opsiyonel, konsolu çok doldurabilir)
    # with ExecutionTimer(f"Poster: {item['title']}"):
    try:
        poster = get_poster_url(item['title'], item['creator'], category)
        item['poster_url'] = poster
    except Exception as e:
        print(f"Poster hatası ({item['title']}): {e}")
    return item

def get_recommendations_from_gemini(user_context, category: Category):
    raw_response_text = ""  # Hata durumunda loglamak için
    try:
        with ExecutionTimer("Prompt Engineering"):
            prompt = build_gemini_prompt(
                category=category,
                age=user_context['age'],
                gender=user_context['gender'],
                emotion=user_context['emotion'],
                raw_scores=user_context['raw_emotion_scores']
            )

        with ExecutionTimer(f"Gemini AI Yanıt ({category.value})"):
            response = model.generate_content(prompt)
            raw_response_text = response.text  # Cevabı sakla

            # Temizlik
            clean_json = raw_response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)

        recommendations = data.get('recommendations', [])

        with ExecutionTimer(f"Poster Arama ({len(recommendations)} Adet)"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(update_item_with_poster, item, category)
                    for item in recommendations
                ]
                concurrent.futures.wait(futures)

        return data

    except json.JSONDecodeError as je:
        print(f"❌ JSON PARSE HATASI!")
        print(f"Gelen Hatalı Metin: {raw_response_text}")  # İşte bunu görmek hayat kurtarır
        return None
    except Exception as e:
        print(f"Genel Servis Hatası: {e}")
        return None