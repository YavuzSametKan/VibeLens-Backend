import os
import ssl
import torch
import cv2
from deepface import DeepFace
from hsemotion.facial_emotions import HSEmotionRecognizer
import numpy as np
import threading

# --- 1. GÜVENLİK AYARLARI ---
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
CAMERA_ID = 1  # Çalışan kameran (0, 1 veya 2)
SKIP_FRAMES = 5  # HIZ SIRRI: Analizi 5 karede bir yap (Daha akıcı olur)

THRESHOLDS = {
    "Happiness": 0.30, "Sadness": 0.30, "Anger": 0.15,
    "Fear": 0.04, "Disgust": 0.05, "Surprise": 0.15, "Contempt": 0.40
}

COLORS = {
    "Happiness": (0, 255, 255), "Sadness": (255, 0, 0),
    "Anger": (0, 0, 255), "Surprise": (255, 165, 0),
    "Fear": (128, 0, 128), "Disgust": (0, 128, 0),
    "Neutral": (200, 200, 200), "Contempt": (255, 255, 0)
}

# --- MODEL ---
try:
    print("⏳ Model Kontrol Ediliyor...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    fer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=device)
    print("✅ Model Hazır!")
except Exception as e:
    fer = None
    print(f"Hata: {e}")

# Global Değişkenler
current_data = {"dominant": "Neutral", "scores": []}
face_coords = None
is_analyzing = False


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
    return best_emotion


def run_analysis(frame):
    global current_data, face_coords, is_analyzing

    if fer is None:
        is_analyzing = False
        return

    try:
        # Resmi Ufalt (Hızlandırır)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Yüz Bul
        face_objs = DeepFace.extract_faces(
            img_path=small_frame,
            detector_backend='retinaface',  # Kaliteli ama yavaş dedektör
            enforce_detection=False,
            align=True
        )

        if not face_objs:
            face_coords = None
            is_analyzing = False
            return

        face_data = face_objs[0]

        # Koordinatları büyüt (0.5 küçültmüştük, 2 ile çarp)
        area = face_data['facial_area']
        face_coords = (area['x'] * 2, area['y'] * 2, area['w'] * 2, area['h'] * 2)

        # Analiz
        face_img = face_data['face']
        face_img_uint8 = (face_img * 255).astype(np.uint8)
        face_img_rgb = cv2.cvtColor(face_img_uint8, cv2.COLOR_BGR2RGB)

        _, scores = fer.predict_emotions(face_img_rgb, logits=False)
        custom_dominant = calculate_custom_emotion(scores)

        current_data = {"dominant": custom_dominant, "scores": scores}

    except Exception:
        pass
    finally:
        is_analyzing = False


def draw_bars(frame, dominant, scores):
    idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness',
                    7: 'Surprise'}
    y_offset = 40
    x_offset = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (240, 280), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, score in enumerate(scores):
        emotion_name = idx_to_class[i]
        percentage = score * 100
        color = COLORS.get(emotion_name, (255, 255, 255))
        threshold = THRESHOLDS.get(emotion_name, 100) * 100

        text_color = (150, 150, 150)
        if emotion_name != "Neutral" and percentage >= threshold:
            text_color = (0, 255, 0)
        if emotion_name == dominant:
            text_color = color

        label = f"{emotion_name[:3]}: {int(percentage)}%"
        cv2.putText(frame, label, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        cv2.rectangle(frame, (x_offset + 90, y_offset - 10), (x_offset + 90 + 100, y_offset + 5), (50, 50, 50), -1)
        fill = int(score * 100)
        cv2.rectangle(frame, (x_offset + 90, y_offset - 10), (x_offset + 90 + fill, y_offset + 5), color, -1)

        if emotion_name != "Neutral":
            tx = x_offset + 90 + int(threshold)
            cv2.line(frame, (tx, y_offset - 12), (tx, y_offset + 7), (255, 255, 255), 1)

        y_offset += 30


def start_camera():
    global is_analyzing
    print(f"🎥 Kamera Başlatılıyor (ID: {CAMERA_ID})...")
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return

    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame_counter += 1

        # --- OPTİMİZASYON: Her 5 karede bir analiz yap ---
        if frame_counter % SKIP_FRAMES == 0 and not is_analyzing:
            is_analyzing = True
            t = threading.Thread(target=run_analysis, args=(frame.copy(),))
            t.daemon = True
            t.start()

        dominant = current_data["dominant"]
        scores = current_data["scores"]

        if face_coords:
            x, y, w, h = face_coords
            color = COLORS.get(dominant, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, dominant.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if len(scores) > 0:
            draw_bars(frame, dominant, scores)

        # Analiz durumu (Sarı nokta = işlemci meşgul)
        status_color = (0, 255, 255) if is_analyzing else (0, 255, 0)
        cv2.circle(frame, (frame.shape[1] - 20, 20), 5, status_color, -1)

        cv2.imshow('VibeLens SMART', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()