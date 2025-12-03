import cv2
import numpy as np
import threading
from deepface import DeepFace
from typing import Dict, Tuple, Optional
from app.core.models import (THRESHOLDS, EMOTION_CLASSES, emotion_recognizer)

# --- 1. CONFIGURATION & CONSTANTS (Only specific to the live demo) ---
CAMERA_ID = 1  # Default camera index (Try 0 if 1 fails)
SCALE_FACTOR = 0.5  # Resize frame for faster processing

# Drawing Colors (BGR format for OpenCV, used in the live test script)
COLORS: Dict[str, Tuple[int, int, int]] = {
    "Happiness": (0, 255, 255), "Sadness": (255, 0, 0),
    "Anger": (0, 0, 255), "Surprise": (0, 165, 255),
    "Fear": (255, 0, 255), "Disgust": (0, 128, 0),
    "Neutral": (200, 200, 200), "Contempt": (255, 255, 0)
}

# --- 2. GLOBAL STATE ---
current_data: Dict = {
    "dominant": "Waiting...",
    "secondary": "None",
    "scores": {}
}
face_coords: Optional[Tuple[int, int, int, int]] = None
is_analyzing: bool = False
lock = threading.Lock()  # Lock for thread-safe state update


# --- 3. EMOTION ALGORITHM (Core VibeLens Logic - Copied for standalone execution) ---
# NOTE: In a clean project, this logic should be imported from vision_service,
# but copying it here makes this test script fully standalone and functional.

def get_secondary_emotion(scores: Dict[str, float], dominant: str) -> str:
    """Identifies the secondary emotion based on raw scores."""
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for emotion, score in sorted_scores:
        if emotion == dominant:
            continue
        if score > 0.01:
            return emotion
    return "None"


def calculate_custom_emotion(scores_array: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """
    Applies the custom threshold-based dynamic scoring algorithm.
    Returns the best emotion and the adjusted (normalized) scores.
    """
    raw_score_dict = {EMOTION_CLASSES[i]: float(s) for i, s in enumerate(scores_array)}
    weighted_scores = {}

    # 1. Calculate Weighted Strength (Raw / Threshold)
    for emotion, raw_val in raw_score_dict.items():
        if emotion == "Neutral":
            weighted_scores[emotion] = raw_val * 0.5
            continue

        threshold = THRESHOLDS.get(emotion, 0.2)

        if raw_val < threshold:
            weighted_scores[emotion] = 0.0
        else:
            weighted_scores[emotion] = raw_val / threshold

    # 2. Determine the Winner
    best_emotion = max(weighted_scores, key=weighted_scores.get)
    final_scores = {}

    # 3. Redistribute Scores (Boost Winner)
    if best_emotion == "Neutral":
        final_scores = raw_score_dict
    else:
        winner_strength = weighted_scores[best_emotion]
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

    # Final normalization step
    total_sum = sum(final_scores.values())
    if total_sum > 0:
        final_scores = {k: v / total_sum for k, v in final_scores.items()}

    return best_emotion, final_scores


# --- 4. ANALYSIS THREAD FUNCTION ---
def run_analysis(frame: np.ndarray):
    """
    Executed in a separate thread to analyze the image frame.
    Updates the global state (current_data, face_coords) using the imported model.
    """
    global current_data, face_coords, is_analyzing

    if emotion_recognizer is None:
        is_analyzing = False
        return

    try:
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        face_objs = DeepFace.extract_faces(
            img_path=small_frame,
            detector_backend='opencv',
            enforce_detection=False,
            align=True
        )

        if not face_objs:
            with lock:
                face_coords = None
            return

        face_data = face_objs[0]

        # Rescale coordinates back to original frame size
        area = face_data['facial_area']
        new_coords = (
            int(area['x'] / SCALE_FACTOR),
            int(area['y'] / SCALE_FACTOR),
            int(area['w'] / SCALE_FACTOR),
            int(area['h'] / SCALE_FACTOR)
        )

        # Prepare face image for HSEmotion
        face_img = face_data['face']
        face_img_uint8 = (face_img * 255).astype(np.uint8)
        face_img_rgb = cv2.cvtColor(face_img_uint8, cv2.COLOR_BGR2RGB)

        # Emotion prediction (Uses imported object)
        _, scores = emotion_recognizer.predict_emotions(face_img_rgb, logits=False)

        # Apply VibeLens custom logic
        dom, adjusted_scores = calculate_custom_emotion(scores)

        raw_score_dict_full = {EMOTION_CLASSES[i]: scores[i] for i in range(len(scores))}
        sec = get_secondary_emotion(raw_score_dict_full, dom)

        # Update global state safely
        with lock:
            current_data = {
                "dominant": dom,
                "secondary": sec,
                "scores": adjusted_scores
            }
            face_coords = new_coords

    except Exception:
        pass
    finally:
        is_analyzing = False


# --- 5. UI DRAWING FUNCTION ---
def draw_ui(frame: np.ndarray):
    """Draws the face bounding box, emotion label, and score panel on the frame."""

    # Safely read global data
    with lock:
        dom = current_data["dominant"]
        sec = current_data["secondary"]
        scores = current_data["scores"]
        coords = face_coords

    # 1. FACE BOX AND LABEL
    if coords:
        x, y, w, h = coords
        # Get color based on dominant emotion (Uses imported COLORS)
        color = COLORS.get(dom, (0, 255, 0))

        # Face Bounding Box and Label Drawing (omitted for brevity, assume content is the same)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{dom.upper()}"
        if sec != "None":
            label += f" ({sec})"
        text_x = x + 5
        text_y = y - 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 3)
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, color, thickness)


    # 2. SCORE PANEL
    if scores:
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        # Panel Background (Black Box)
        cv2.rectangle(frame, (10, 10), (270, 320), (0, 0, 0), -1)
        y_offset = 40

        for emotion, score in sorted_scores:
            pct = int(score * 100)
            color = COLORS.get(emotion, (255, 255, 255))

            # Highlight dominant emotion
            text_color = (180, 180, 180)
            if emotion == dom: text_color = color

            # Text (Emotion Name: Percentage)
            text = f"{emotion}: {pct}%"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Bar Visualization
            bar_len = int(pct * 1.3)
            cv2.rectangle(frame, (155, y_offset - 10), (155 + bar_len, y_offset), color, -1)
            y_offset += 30


# --- 6. MAIN CAMERA LOOP ---
def start_camera():
    """Initializes the camera and runs the main video processing loop."""
    global is_analyzing
    print(f" Starting camera capture (ID: {CAMERA_ID})...")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(" Could not open camera.")
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

        cv2.imshow('VibeLens LIVE Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()