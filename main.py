from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from models import Category, VibeResponse
from services import analyze_image_with_deepface, get_recommendations_from_gemini

app = FastAPI(title="VibeLens API")


@app.post("/analyze", response_model=VibeResponse)
async def analyze(
        category: Category = Form(...),
        file: UploadFile = File(...)
):
    # 1. Görüntüyü İşle
    image_bytes = await file.read()
    user_context = analyze_image_with_deepface(image_bytes)

    if not user_context:
        raise HTTPException(status_code=400, detail="Yüz tespit edilemedi.")

    # 2. Öneri Al
    recommendation_data = get_recommendations_from_gemini(user_context, category)

    if not recommendation_data:
        raise HTTPException(status_code=500, detail="Yapay zeka yanıt veremedi.")

    # 3. Tüm Veriyi Birleştirip Dön (Response Model'e Uygun Hale Getir)
    return VibeResponse(
        # AI'dan gelen başlık ve açıklamalar
        mood_title=recommendation_data['mood_title'],
        mood_description=recommendation_data['mood_description'],
        recommendations=recommendation_data['recommendations'],

        # DeepFace'ten gelen analiz verileri
        dominant_emotion=user_context['emotion'],
        detected_age=user_context['age'],
        detected_gender=user_context['gender'],
        emotion_scores=user_context['raw_emotion_scores']
    )