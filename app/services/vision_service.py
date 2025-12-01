import cv2
import numpy as np
import ssl
import torch
import os
from deepface import DeepFace
from hsemotion.facial_emotions import HSEmotionRecognizer
from app.utils.timer import ExecutionTimer

# --- AYARLAR ---
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

# --- MODEL BAŞLATMA ---
print("⏳ Vision Modelleri Hazırlanıyor...")
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