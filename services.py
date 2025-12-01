import cv2
import numpy as np
import json
import os
import ssl
import torch
import concurrent.futures
from deepface import DeepFace
# HSEmotion Importu
from hsemotion.facial_emotions import HSEmotionRecognizer
import google.generativeai as genai
from models import Category
from prompts import build_gemini_prompt
from search_service import get_poster_url
from utils import ExecutionTimer
from dotenv import load_dotenv

# --- 1. SİSTEM VE GÜVENLİK AYARLARI (SSL & PyTorch Fix) ---
os.environ['CURL_CA_BUNDLE'] = ''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# PyTorch 2.6+ Güvenlik Bypass (Modeli yükleyebilmek için)
_original_torch_load = torch.load


def _unsafe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _unsafe_torch_load

# --- 2. KONFİGÜRASYON ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    'gemini-flash-latest',
    generation_config={"response_mime_type": "application/json"}
)

# --- 3. HSEMOTION MODELİ YÜKLEME ---
print("⏳ AI Modelleri Hazırlanıyor...")
try:
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # Testte kullandığımız en iyi model
    fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
    print(f"✅ HSEmotion (Duygu) Hazır! ({device})")
except Exception as e:
    print(f"❌ HSEmotion Hatası: {e}")
    fer = None

# --- 4. HASSAS EŞİK AYARLARI (Live Test'ten Birebir) ---
THRESHOLDS = {
    "Happiness": 0.30,
    "Sadness": 0.30,
    "Anger": 0.15,
    "Fear": 0.04,  # %4 Korku görse bile yakalayacak
    "Disgust": 0.05,
    "Surprise": 0.15,
    "Contempt": 0.40
}


# --- YARDIMCI FONKSİYONLAR ---

def calculate_custom_emotion(scores):
    """
    HSEmotion skorlarını alıp, bizim hassas eşiklerimize göre karar verir.
    """
    idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness',
                    7: 'Surprise'}

    # Skor listesini sözlüğe çevir: {'Anger': 0.05, ...}
    score_dict = {idx_to_class[i]: s for i, s in enumerate(scores)}

    # 1. Öncelik: Neutral hariç, eşiği geçen en yüksek skoru bul
    best_emotion = "Neutral"
    max_score = 0

    for emotion, score in score_dict.items():
        if emotion == "Neutral": continue

        limit = THRESHOLDS.get(emotion, 0.2)
        if score > limit and score > max_score:
            max_score = score
            best_emotion = emotion

    # Eğer custom bir duygu bulunamadıysa Neutral dön, bulunduysa onu dön
    return best_emotion, score_dict


def analyze_image_with_smart_ai(image_bytes):
    """
    1. RetinaFace ile yüzü bul.
    2. DeepFace ile Yaş/Cinsiyet al.
    3. HSEmotion ile Hassas Duygu Analizi yap.
    """
    with ExecutionTimer("Gelişmiş Görüntü Analizi"):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 1. Adım: Yüzü Bul ve Yaş/Cinsiyet Analizi (DeepFace)
            # RetinaFace kullanarak yüzü buluyoruz
            demography_objs = DeepFace.analyze(
                img_path=img,
                actions=['age', 'gender'],  # Sadece yaş ve cinsiyet soruyoruz
                detector_backend='retinaface',
                enforce_detection=False,
                silent=True
            )

            demography = demography_objs[0] if isinstance(demography_objs, list) else demography_objs

            # Yüz koordinatlarını al (HSEmotion için yüzü keseceğiz)
            region = demography['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Resmi kes (Crop)
            face_img = img[y:y + h, x:x + w]

            # 2. Adım: HSEmotion için hazırlık
            # Eğer yüz çok küçükse veya bulunamadıysa DeepFace verisini kullan (Fallback)
            if face_img.size == 0:
                print("⚠️ Yüz kesilemedi, varsayılan analiz kullanılıyor.")
                return None

            # HSEmotion formatı (RGB ve Resize)
            face_img = cv2.resize(face_img, (224, 224))  # HSEmotion 224x224 sever
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # 3. Adım: HSEmotion Tahmini
            if fer:
                _, scores = fer.predict_emotions(face_img_rgb, logits=False)
                dominant_emotion, score_dict = calculate_custom_emotion(scores)
            else:
                # Eğer HSEmotion yüklenemezse DeepFace'e dön (Yedek)
                dominant_emotion = "Neutral (Backup)"
                score_dict = {}

            # Yaş ve Cinsiyeti formatla
            age = int(demography['age'])
            gender = demography['dominant_gender']

            return {
                "emotion": dominant_emotion,  # Artık "Sadness", "Fear" gibi hassas sonuçlar
                "age": age,
                "gender": gender,
                "raw_emotion_scores": {k: float(v) for k, v in score_dict.items()}  # JSON için float dönüşümü
            }

        except Exception as e:
            print(f"Vision Pipeline Hatası: {e}")
            return None


def update_item_with_poster(item, category):
    try:
        poster = get_poster_url(item['title'], item['creator'], category)
        item['poster_url'] = poster
    except Exception as e:
        print(f"Poster hatası: {e}")
    return item


def get_recommendations_from_gemini(user_context, category: Category):
    try:
        with ExecutionTimer("Prompt Engineering"):
            prompt = build_gemini_prompt(
                category=category,
                age=user_context['age'],
                gender=user_context['gender'],
                emotion=user_context['emotion'],
                raw_scores=user_context['raw_emotion_scores']
            )

        with ExecutionTimer(f"Gemini AI ({category.value})"):
            response = model.generate_content(prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
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

    except Exception as e:
        print(f"AI Servis Hatası: {e}")
        return None