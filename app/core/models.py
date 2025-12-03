import os
import ssl
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer
from typing import Dict

# --- 1. SECURITY AND COMPATIBILITY FIXES ---

# Fix for OpenSSL certificate verification errors (often needed in dev environments)
# This disables strict SSL certificate verification for model downloads.
os.environ['CURL_CA_BUNDLE'] = ''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# PyTorch Compatibility Fix: Forces weights_only=False for loading older models securely
_original_torch_load = torch.load
def _unsafe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _unsafe_torch_load

# --- 2. CONFIGURATION & CONSTANTS ---

# Custom Emotion Thresholds (Critical for the VibeLens analysis algorithm)
THRESHOLDS: Dict[str, float] = {
    "Happiness": 0.30, "Sadness": 0.25, "Anger": 0.12,
    "Fear": 0.035, "Disgust": 0.15, "Surprise": 0.14, "Contempt": 0.47
}

# Emotion classes mapping for raw model output
EMOTION_CLASSES: Dict[int, str] = {
    0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear',
    4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'
}

# --- 3. MODEL INITIALIZATION ---
print(" Preparing Vision Models...")
try:
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # The emotion_recognizer object will be imported by service files
    emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device=DEVICE)
    print(f" HSEmotion Ready! ({DEVICE})")
except Exception as e:
    print(f" HSEmotion Error: {e}")
    emotion_recognizer = None