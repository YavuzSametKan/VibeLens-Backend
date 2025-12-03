import cv2
import numpy as np
import torch
from deepface import DeepFace
from hsemotion.facial_emotions import HSEmotionRecognizer

# --- PyTorch Compatibility Fix (See explanation below) ---
_original_torch_load = torch.load


def _unsafe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _unsafe_torch_load

# --- CONFIGURATION ---
THRESHOLDS = {
    "Happiness": 0.30, "Sadness": 0.25, "Anger": 0.12,
    "Fear": 0.035, "Disgust": 0.15, "Surprise": 0.14, "Contempt": 0.47
}

EMOTION_CLASSES = {
    0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear',
    4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'
}

# --- MODEL INITIALIZATION ---
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
try:
    emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=DEVICE)
except Exception:
    emotion_recognizer = None


# --- HELPER FUNCTIONS ---
def get_secondary_emotion(scores: dict, dominant: str) -> str:
    """Identifies the secondary emotion based on raw scores."""
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for emotion, score in sorted_scores:
        if emotion == dominant:
            continue
        if score > 0.01:
            return emotion
    return "None"


def calculate_custom_emotion(raw_scores: np.ndarray) -> tuple[str, dict]:
    """
    Dynamic Scoring Algorithm: Calculates 'Relative Strength' (Raw_Score / Threshold)
    and normalizes final scores to emphasize the winning emotion.
    """
    raw_score_dict = {EMOTION_CLASSES[i]: float(s) for i, s in enumerate(raw_scores)}
    weighted_scores = {}

    # 1. Calculate Weighted Strength
    for emotion, raw_val in raw_score_dict.items():
        if emotion == "Neutral":
            weighted_scores[emotion] = raw_val * 0.5
            continue

        threshold = THRESHOLDS.get(emotion, 0.2)
        weighted_scores[emotion] = raw_val / threshold if raw_val >= threshold else 0.0

    # 2. Determine the Winner
    best_emotion = max(weighted_scores, key=weighted_scores.get)

    # 3. Redistribute Scores (Normalization)
    final_scores = {}

    if best_emotion == "Neutral":
        final_scores = raw_score_dict
    else:
        winner_strength = weighted_scores[best_emotion]
        new_winner_score = min(0.50 + (winner_strength * 0.1), 0.90)

        final_scores[best_emotion] = new_winner_score
        remaining_pie = 1.0 - new_winner_score

        raw_others_sum = sum([v for k, v in raw_score_dict.items() if k != best_emotion])

        for emo, val in raw_score_dict.items():
            if emo != best_emotion:
                final_scores[emo] = (val / raw_others_sum) * remaining_pie if raw_others_sum > 0 else 0.0

    # Final normalization step
    total_sum = sum(final_scores.values())
    if total_sum > 0:
        final_scores = {k: v / total_sum for k, v in final_scores.items()}

    return best_emotion, final_scores


# --- MAIN ANALYSIS FUNCTION ---
def analyze_image_with_smart_ai(image_bytes: bytes) -> dict | None:
    """Performs multi-step analysis (Demography + Custom Emotion Scoring) on an image."""
    try:
        # Decode Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # DeepFace Demography Analysis (Age, Gender, Face Region)
        demography_objs = DeepFace.analyze(
            img_path=img,
            actions=['age', 'gender'],
            detector_backend='retinaface',
            enforce_detection=False,
            silent=True
        )
        demography = demography_objs[0]

        region = demography['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # Prepare Face Image for Emotion Recognition
        face_img = img[y:y + h, x:x + w]
        if face_img.size == 0:
            return None

        face_img = cv2.resize(face_img, (224, 224))
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # HSEmotion Prediction
        _, raw_scores = emotion_recognizer.predict_emotions(face_img_rgb, logits=False)

        # Custom Emotion Scoring
        dominant_emotion, adjusted_score_dict = calculate_custom_emotion(raw_scores)

        # Calculate Secondary Emotion
        raw_score_dict_full = {EMOTION_CLASSES[i]: raw_scores[i] for i in range(len(raw_scores))}
        secondary_emotion = get_secondary_emotion(raw_score_dict_full, dominant_emotion)

        # Final Result Assembly
        return {
            "emotion": dominant_emotion,
            "secondary_emotion": secondary_emotion,
            "age": int(demography['age']),
            "gender": demography['dominant_gender'],
            "raw_emotion_scores": dict(sorted(
                adjusted_score_dict.items(),
                key=lambda item: item[1],
                reverse=True
            ))
        }

    except Exception:
        return None