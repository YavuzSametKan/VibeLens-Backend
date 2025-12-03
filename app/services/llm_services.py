import json
import concurrent.futures
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from app.core.config import settings
from app.schemas.analysis import Category
from app.utils.timer import ExecutionTimer
from app.core.prompts import build_gemini_prompt

from app.services.search_service import get_content_metadata

# Gemini Config
genai.configure(api_key=settings.GEMINI_API_KEY)

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name = 'gemini-flash-latest',
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.9,       # YENİ: Yaratıcılığı artırdık (Eskiden 0.7 veya varsayılandı)
        "top_p": 0.95,            # YENİ: Kelime havuzunu genişlettik
    },
    safety_settings = safety_settings
)

# --- AYARLAR ---
MAX_RETRIES = 3  # En fazla 3 kere dene
RETRY_DELAY = 2 # Her hatada 2 saniye bekle

def update_item_with_metadata(item, category):
    """
    Gemini verisi ile API verisini akıllıca birleştirir.
    """
    try:
        metadata = get_content_metadata(item['title'], item['creator'], category)

        # Poster: Her zaman API'den al
        item['poster_url'] = metadata['poster']

        # Diğer verileri API doluysa güncelle
        if metadata.get('overview'):
            item['overview'] = metadata['overview']
        if metadata.get('rating'):
            item['rating'] = metadata['rating']
        if metadata.get('year'):
            item['year'] = metadata['year']
        if metadata.get('external_links'):
            item['external_links'] = metadata['external_links']

    except Exception as e:
        print(f"Metadata merge hatası ({item.get('title')}): {e}")
    return item

def get_fallback_response():
    """
    Eğer AI tamamen çökerse, uygulama patlamasın diye dönülecek varsayılan veri.
    """
    return {
        "mood_title": "Bağlantı Sorunu",
        "mood_description": "Yapay zeka şu an biraz yoğun, ancak senin için rastgele popüler içerikler getirebilirim.",
        "recommendations": [] # Boş liste döneriz, frontend bunu "Öneri Yok" diye gösterir
    }


def get_recommendations_from_gemini(user_context, category: Category):
    # Prompt Hazırlığı
    try:
        dominant = user_context['emotion']
        raw_scores = user_context['raw_emotion_scores']
        secondary = user_context['secondary_emotion']

        prompt = build_gemini_prompt(
            category=category,
            age=user_context['age'],
            gender=user_context['gender'],
            emotion=dominant,
            secondary_emotion=secondary,
            raw_scores=raw_scores
        )
    except Exception as e:
        print(f"Prompt Hatası: {e}")
        return get_fallback_response()

    # --- RETRY MEKANİZMASI ---
    data = None
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with ExecutionTimer(f"Gemini AI ({category.value}) - Deneme {attempt}/{MAX_RETRIES}"):
                response = model.generate_content(prompt)

                # Cevabı almayı dene
                try:
                    raw_text = response.text
                except ValueError:
                    # Boş geldiyse hata fırlat ki except bloğuna düşsün ve retry yapsın
                    print(f"⚠️ Deneme {attempt}: Gemini boş yanıt döndü. Sebebi: {response.prompt_feedback}")
                    raise ValueError("Empty Response from Gemini")

                clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_json)

                # Eğer buraya geldiyse başarıldı demektir, döngüyü kır
                break

        except Exception as e:
            print(f"❌ Deneme {attempt} Başarısız: {e}")
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"⏳ {RETRY_DELAY} saniye bekleniyor...")
                time.sleep(RETRY_DELAY)
            else:
                print("🚨 Tüm denemeler başarısız oldu.")

    # Eğer tüm denemelerden sonra data hala yoksa Fallback dön
    if not data:
        print("⚠️ Fallback (Acil Durum) verisi dönülüyor.")
        return get_fallback_response()

    # --- METADATA İŞLEMLERİ (Sadece data varsa yapılır) ---
    try:
        recommendations = data.get('recommendations', [])

        if recommendations:
            with ExecutionTimer(f"Metadata Zenginleştirme ({len(recommendations)} Adet)"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(update_item_with_metadata, item, category)
                        for item in recommendations
                    ]
                    concurrent.futures.wait(futures)

        return data

    except Exception as e:
        print(f"Metadata Süreç Hatası: {e}")
        # Metadata patlasa bile çıplak datayı dönelim, hiç yoktan iyidir
        return data