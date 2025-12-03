import cv2
import numpy as np
import torch
from deepface import DeepFace
from hsemotion.facial_emotions import HSEmotionRecognizer
from app.utils.timer import ExecutionTimer

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
    print(f" HSEmotion Hazır! ({device})")
except Exception as e:
    print(f" HSEmotion Hatası: {e}")
    fer = None

THRESHOLDS = {
    "Happiness": 0.30, "Sadness": 0.25, "Anger": 0.12,
    "Fear": 0.035, "Disgust": 0.15, "Surprise": 0.14, "Contempt": 0.47
}

def get_secondary_emotion(scores, dominant):
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for emotion, score in sorted_scores:
        if emotion == dominant:
            continue
        if score > 0.01:
            return emotion
    return "None"

def calculate_custom_emotion(scores):
    """
        DİNAMİK SKORLAMA ALGORİTMASI:
        Her duygunun kendi eşiğine göre 'Göreceli Gücünü' (Relative Strength) hesaplar.

        Formül: Strength = Ham_Skor / Eşik_Değeri
        Örn: Fear (%5) / Eşik (%4) = 1.25 Güç (Baskın!)
             Happy (%30) / Eşik (%30) = 1.0 Güç
    """
    idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear',
                    4: 'Happiness', 5: 'Neutral', 6: 'Sadness', 7: 'Surprise'}
    raw_score_dict = {idx_to_class[i]: float(s) for i, s in enumerate(scores)}
    score_dict = {idx_to_class[i]: s for i, s in enumerate(scores)}
    weighted_scores = {}

    for emotion, raw_val in raw_score_dict.items():
        if emotion == "Neutral":
            # Neutral'in eşiği yok, onu baz puan olarak alıyoruz ama katsayısını düşük tutuyoruz
            weighted_scores[emotion] = raw_val * 0.5
            continue

        threshold = THRESHOLDS.get(emotion, 0.2)

        # Eğer ham skor eşiğin altındaysa, gücü 0 kabul et (Elendi)
        if raw_val < threshold:
            weighted_scores[emotion] = 0.0
        else:
            # Eşiği geçtiyse: Skor / Eşik oranıyla gücünü belirle
            # Fear 0.05 / 0.04 = 1.25 puan alır.
            # Happy 0.35 / 0.30 = 1.16 puan alır.
            weighted_scores[emotion] = raw_val / threshold

        # 3. Kazananı Belirle (En yüksek ağırlıklı puana sahip olan)
        # Eğer hiçbir duygu eşiği geçemediyse (hepsi 0.0) -> Neutral kazanır.
    best_emotion = max(weighted_scores, key=weighted_scores.get)

    # Eğer kazanan Neutral değilse ama puanı çok düşükse yine de Neutral'e dönmesin,
    # weighted_scores mantığı zaten elemeyi yaptı.

    # 4. SKORLARI YENİDEN DAĞIT (Softmax Benzeri Normalizasyon)
    # Kazanan duygunun hakkını vermek için skorları Gemini'ye uygun hale getiriyoruz.
    # Kazanan duyguya "Ağırlığı kadar" pay verip, kalanları orantılı dağıtıyoruz.

    final_scores = {}

    if best_emotion == "Neutral":
        # Eğer kazanan Neutral ise, ham veriyi olduğu gibi (veya hafif sadeleştirerek) döndür
        final_scores = raw_score_dict
    else:
        # Kazanan bir duygu var (Örn: Fear).
        # Onu belirginleştirmek için yapay bir "Güven Skoru" oluşturuyoruz.
        # Sabit %60 yerine, kendi gücüne dayalı bir artış.

        # Kazananın gücü (Örn: 1.25)
        winner_strength = weighted_scores[best_emotion]

        # Yeni Puan = 0.50 + (Güç * 0.1) -> Min %50, Güç arttıkça artar, Max %90'da keseriz.
        new_winner_score = min(0.50 + (winner_strength * 0.1), 0.90)

        final_scores[best_emotion] = new_winner_score
        remaining_pie = 1.0 - new_winner_score

        # Kalan payı diğerlerine (Neutral dahil) eski oranlarına göre dağıt
        raw_others_sum = sum([v for k, v in raw_score_dict.items() if k != best_emotion])

        for emo, val in raw_score_dict.items():
            if emo == best_emotion: continue

            if raw_others_sum > 0:
                final_scores[emo] = (val / raw_others_sum) * remaining_pie
            else:
                final_scores[emo] = 0.0

    return best_emotion, final_scores

def analyze_image_with_smart_ai(image_bytes):
    with ExecutionTimer("Gelişmiş Görüntü Analizi"):
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            demography_objs = DeepFace.analyze(
                img_path=img,
                actions=['age', 'gender'],
                detector_backend='retinaface',
                enforce_detection=False,
                silent=True
            )
            demography = demography_objs[0] if isinstance(demography_objs, list) else demography_objs

            region = demography['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            face_img = img[y:y + h, x:x + w]
            if face_img.size == 0: return None

            face_img = cv2.resize(face_img, (224, 224))
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            _, scores = fer.predict_emotions(face_img_rgb, logits=False)
            dominant_emotion, adjusted_score_dict = calculate_custom_emotion(scores)
            score_dict = {
                'Anger': scores[0], 'Contempt': scores[1], 'Disgust': scores[2],
                'Fear': scores[3], 'Happiness': scores[4], 'Neutral': scores[5],
                'Sadness': scores[6], 'Surprise': scores[7]
            }
            secondary_emotion = get_secondary_emotion(score_dict, dominant_emotion)

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
        except Exception as e:
            print(f"Vision Pipeline Hatası: {e}")
            return None