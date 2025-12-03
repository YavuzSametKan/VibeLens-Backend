import os
import ssl
import sys
import torch
import cv2
import numpy as np
import threading
import time
from deepface import DeepFace
from hsemotion.facial_emotions import HSEmotionRecognizer

# --- 1. GÜVENLİK VE FIXLER ---
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

# --- 2. AYARLAR ---
CAMERA_ID = 1  # Açılmazsa 1 yap
SCALE_FACTOR = 0.5  # Hız için resmi yarıya indiriyoruz (OpenCV ile uçar)

# SENİN BELİRLEDİĞİN BİREBİR EŞİKLER
THRESHOLDS = {
    "Happiness": 0.30, "Sadness": 0.25, "Anger": 0.12,
    "Fear": 0.035, "Disgust": 0.15, "Surprise": 0.14, "Contempt": 0.47
}

COLORS = {
    "Happiness": (0, 255, 255), "Sadness": (255, 0, 0),
    "Anger": (0, 0, 255), "Surprise": (0, 165, 255),
    "Fear": (255, 0, 255), "Disgust": (0, 128, 0),
    "Neutral": (200, 200, 200), "Contempt": (255, 255, 0)
}

# --- MODEL ---
print("⏳ Vision Modelleri Hazırlanıyor...")
try:
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
    print(f"✅ HSEmotion Hazır! ({device})")
except Exception as e:
    print(f"❌ HSEmotion Hatası: {e}")
    fer = None

current_data = {
    "dominant": "Waiting...",
    "secondary": "None",
    "scores": {}
}
face_coords = None
is_analyzing = False


# --- SENİN MATEMATİKSEL MANTIĞIN (BİREBİR) ---

def get_secondary_emotion(scores, dominant):
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for emotion, score in sorted_scores:
        if emotion == dominant:
            continue
        if score > 0.01:
            return emotion
    return "None"


def calculate_custom_emotion(scores):
    idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness',
                    7: 'Surprise'}
    raw_score_dict = {idx_to_class[i]: float(s) for i, s in enumerate(scores)}

    weighted_scores = {}

    for emotion, raw_val in raw_score_dict.items():
        if emotion == "Neutral":
            # Neutral cezası (Senin kodun)
            weighted_scores[emotion] = raw_val * 0.5
            continue

        threshold = THRESHOLDS.get(emotion, 0.2)

        if raw_val < threshold:
            weighted_scores[emotion] = 0.0
        else:
            weighted_scores[emotion] = raw_val / threshold

    best_emotion = max(weighted_scores, key=weighted_scores.get)

    final_scores = {}

    if best_emotion == "Neutral":
        final_scores = raw_score_dict
    else:
        winner_strength = weighted_scores[best_emotion]
        # Senin Boost Formülün
        new_winner_score = min(0.50 + (winner_strength * 0.1), 0.90)

        final_scores[best_emotion] = new_winner_score
        remaining_pie = 1.0 - new_winner_score

        raw_others_sum = sum([v for k, v in raw_score_dict.items() if k != best_emotion])

        for emo, val in raw_score_dict.items():
            if emo == best_emotion: continue

            if raw_others_sum > 0:
                final_scores[emo] = (val / raw_others_sum) * remaining_pie
            else:
                final_scores[emo] = 0.0

    return best_emotion, final_scores


def run_analysis(frame):
    global current_data, face_coords, is_analyzing

    if fer is None:
        is_analyzing = False
        return

    try:
        # Hız için küçültme (OpenCV ile birleşince uçar)
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        # --- DEĞİŞİKLİK BURADA: detector_backend='opencv' ---
        # RetinaFace yerine OpenCV kullanıyoruz. Çok hızlıdır.
        face_objs = DeepFace.extract_faces(
            img_path=small_frame,
            detector_backend='opencv',
            enforce_detection=False,
            align=True
        )

        if not face_objs:
            face_coords = None
            is_analyzing = False
            return

        face_data = face_objs[0]

        # Koordinatları geri büyüt
        area = face_data['facial_area']
        face_coords = (
            int(area['x'] / SCALE_FACTOR),
            int(area['y'] / SCALE_FACTOR),
            int(area['w'] / SCALE_FACTOR),
            int(area['h'] / SCALE_FACTOR)
        )

        face_img = face_data['face']
        face_img_uint8 = (face_img * 255).astype(np.uint8)
        face_img_rgb = cv2.cvtColor(face_img_uint8, cv2.COLOR_BGR2RGB)

        _, scores = fer.predict_emotions(face_img_rgb, logits=False)

        # Senin mantığını çalıştır
        dom, adjusted_scores = calculate_custom_emotion(scores)

        raw_score_dict = {
            'Anger': scores[0], 'Contempt': scores[1], 'Disgust': scores[2],
            'Fear': scores[3], 'Happiness': scores[4], 'Neutral': scores[5],
            'Sadness': scores[6], 'Surprise': scores[7]
        }
        sec = get_secondary_emotion(raw_score_dict, dom)

        current_data = {
            "dominant": dom,
            "secondary": sec,
            "scores": adjusted_scores
        }

    except Exception:
        pass
    finally:
        is_analyzing = False

def draw_ui(frame):
    dom = current_data["dominant"]
    sec = current_data["secondary"]
    scores = current_data["scores"]

    # 1. YÜZ KUTUSU VE ETİKETİ
    if face_coords:
        x, y, w, h = face_coords
        # Duygunun kendi rengini al (Yoksa yeşil yap)
        color = COLORS.get(dom, (0, 255, 0))

        # Yüz Çerçevesi (Biraz daha ince ve zarif)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Etiket Metni
        label = f"{dom.upper()}"
        if sec != "None":
            label += f" ({sec})"

        # --- YENİ: KONTURLU VE BÜYÜK METİN ---
        # Konum: Yüzün biraz üstü
        text_x = x + 5
        text_y = y - 15

        # Font ayarları (Daha büyük)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2  # Eskiden 0.6 idi, büyüttük
        thickness = 3

        # ADIM 1: Siyah Çerçeve (Kalın siyah metin)
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 3)

        # ADIM 2: Renkli Metin (Üzerine ince renkli metin)
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, color, thickness)

    # 2. SKOR PANELİ (Aynı kalıyor, sadece fontu biraz büyüttüm)
    if scores:
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Panel Arka Planı (Biraz daha genişlettim)
        cv2.rectangle(frame, (10, 10), (270, 320), (0, 0, 0), -1)
        y = 40

        for emotion, score in sorted_scores:
            pct = int(score * 100)
            color = COLORS.get(emotion, (255, 255, 255))

            # Seçili olan parlak, diğerleri gri
            text_color = (180, 180, 180)
            if emotion == dom: text_color = color

            # Metin (Daha okunaklı font)
            text = f"{emotion}: {pct}%"
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Bar
            bar_len = int(pct * 1.3)  # Barı biraz uzattım
            cv2.rectangle(frame, (155, y - 10), (155 + bar_len, y), color, -1)
            y += 30


def start_camera():
    global is_analyzing
    print(f"🎥 Kamera Başlatılıyor (ID: {CAMERA_ID})...")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("❌ Kamera açılamadı.")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        if not is_analyzing:
            is_analyzing = True
            t = threading.Thread(target=run_analysis, args=(frame.copy(),))
            t.daemon = True
            t.start()

        draw_ui(frame)
        cv2.imshow('VibeLens LIVE (OpenCV)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()