from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.analysis import Category, VibeResponse
from app.services.vision_service import analyze_image_with_smart_ai
from app.services.llm_services import get_recommendations_from_gemini

router = APIRouter()

@router.post("/analyze", response_model=VibeResponse)
async def analyze(
        category: Category = Form(...),
        file: UploadFile = File(...)
):
    # 1. Görüntüyü İşle
    image_bytes = await file.read()
    user_context = analyze_image_with_smart_ai(image_bytes)

    if not user_context:
        raise HTTPException(status_code=400, detail="Yüz tespit edilemedi.")

    # 2. Öneri Al
    recommendation_data = get_recommendations_from_gemini(user_context, category)

    if not recommendation_data:
        raise HTTPException(status_code=500, detail="Yapay zeka yanıt veremedi.")

    # 3. Yanıt Oluştur
    return VibeResponse(
        mood_title=recommendation_data['mood_title'],
        mood_description=recommendation_data['mood_description'],
        recommendations=recommendation_data['recommendations'],
        dominant_emotion=user_context['emotion'],
        secondary_emotion=user_context['secondary_emotion'],
        detected_age=user_context['age'],
        detected_gender=user_context['gender'],
        emotion_scores=user_context['raw_emotion_scores']
    )