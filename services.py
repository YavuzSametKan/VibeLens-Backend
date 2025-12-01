import cv2
import numpy as np
import json
import os
import ssl
import torch
import concurrent.futures
from deepface import DeepFace
from hsemotion.facial_emotions import HSEmotionRecognizer
import google.generativeai as genai
from models import Category
from app.core.prompts import build_gemini_prompt
from search_service import get_content_metadata  # Metadata fonksiyonunu kullanıyoruz
from utils import ExecutionTimer
from dotenv import load_dotenv

# --- AYARLAR (Aynı) ---
os.environ['CURL_CA_BUNDLE'] = ''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

_original_torch_load = torch.load


def _unsafe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _unsafe_torch_load

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    'gemini-flash-latest',
    generation_config={"response_mime_type": "application/json"}
)

print("⏳ AI Modelleri Hazırlanıyor...")
try:
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
    print(f"✅ HSEmotion Hazır! ({device})")
except Exception as e:
    print(f"❌ HSEmotion Hatası: {e}")
    fer = None

THRESHOLDS = {
    "Happiness": 0.30, "Sadness": 0.30, "Anger": 0.15,
    "Fear": 0.04, "Disgust": 0.05, "Surprise": 0.15, "Contempt": 0.40
}


def calculate_custom_emotion(scores):
    idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness',
                    7: 'Surprise'}
    score_dict = {idx_to_class[i]: s for i, s in enumerate(scores)}
    best_emotion = "Neutral"
    max_score = 0
    for emotion, score in score_dict.items():
        if emotion == "Neutral": continue
        limit = THRESHOLDS.get(emotion, 0.2)
        if score > limit and score > max_score:
            max_score = score
            best_emotion = emotion
    return best_emotion, score_dict


def analyze_image_with_smart_ai(image_bytes):
    # ... (Bu fonksiyon ZATEN DOĞRU çalışıyor, değiştirmeye gerek yok) ...
    # (Yukarıdaki koddan aynen kopyala veya mevcut olan kalsın)
    # Kısaltmak için burayı atlıyorum, çünkü sorunumuz metadata'da.
    with ExecutionTimer("Gelişmiş Görüntü Analizi"):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            demography_objs = DeepFace.analyze(img_path=img, actions=['age', 'gender'], detector_backend='retinaface',
                                               enforce_detection=False, silent=True)
            demography = demography_objs[0] if isinstance(demography_objs, list) else demography_objs
            region = demography['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            face_img = img[y:y + h, x:x + w]
            if face_img.size == 0: return None
            face_img = cv2.resize(face_img, (224, 224))
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            if fer:
                _, scores = fer.predict_emotions(face_img_rgb, logits=False)
                dominant_emotion, score_dict = calculate_custom_emotion(scores)
            else:
                dominant_emotion = "Neutral (Backup)"
                score_dict = {}
            return {
                "emotion": dominant_emotion,
                "age": int(demography['age']),
                "gender": demography['dominant_gender'],
                "raw_emotion_scores": {k: float(v) for k, v in score_dict.items()}
            }
        except Exception as e:
            print(f"Vision Pipeline Hatası: {e}")
            return None


# --- KRİTİK BÖLÜM: METADATA BİRLEŞTİRME ---

def update_item_with_metadata(item, category):
    """
    Gemini verisi ile API verisini akıllıca birleştirir.
    """
    try:
        # API'den veriyi çek (Kitap/Müzik için sadece poster, diğerleri None gelir)
        metadata = get_content_metadata(item['title'], item['creator'], category)

        # 1. Poster: Her zaman API'den geleni al (Placeholder olsa bile)
        item['poster_url'] = metadata['poster']

        # 2. Diğer Veriler: Sadece API dolu gönderdiyse güncelle!
        # Eğer API'den None geldiyse, Gemini'nin yazdığı değer item içinde kalır.

        if metadata.get('overview') is not None and metadata.get('overview') != "":
            item['overview'] = metadata['overview']

        if metadata.get('rating') is not None and metadata.get('rating') != "":
            item['rating'] = metadata['rating']

        if metadata.get('year') is not None and metadata.get('year') != "":
            item['year'] = metadata['year']

        if metadata.get('external_links'):
            item['external_links'] = metadata['external_links']

    except Exception as e:
        print(f"Metadata merge hatası ({item['title']}): {e}")
    return item


def get_secondary_emotion(scores, dominant):
    """
    Baskın duygu ve 'Neutral' haricindeki en yüksek ikinci duyguyu bulur.
    Örn: Sadness (%40) -> Baskın. Sırada Anger (%10) varsa -> İkincil.
    """
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    for emotion, score in sorted_scores:
        # Baskın olanı ve Neutral'i atla (Neutral genelde baz puandır)
        if emotion == dominant or emotion == "Neutral":
            continue
        # Eğer kayda değer bir skor varsa (%1 bile olsa) döndür
        if score > 0.01:
            return emotion

    return "None"

def get_recommendations_from_gemini(user_context, category: Category):
    try:
        # İkincil duyguyu hesapla
        dominant = user_context['emotion']
        raw_scores = user_context['raw_emotion_scores']
        secondary = get_secondary_emotion(raw_scores, dominant)

        with ExecutionTimer("Prompt Engineering"):
            prompt = build_gemini_prompt(
                category=category,
                age=user_context['age'],
                gender=user_context['gender'],
                emotion=dominant,
                secondary_emotion=secondary,  # YENİ PARAMETRE
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
                    # Fonksiyonumuz artık update_item_with_metadata
                    executor.submit(update_item_with_metadata, item, category)
                    for item in recommendations
                ]
                concurrent.futures.wait(futures)

        return data

    except Exception as e:
        print(f"AI Servis Hatası: {e}")
        return None