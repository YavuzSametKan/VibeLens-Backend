from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.schemas.analysis import Category, VibeResponse
from app.services.vision_service import analyze_image_with_smart_ai
from app.services.llm_services import get_recommendations_from_gemini

ROOT_DIR = Path(__file__).parent.parent.parent
STATUS_HTML_FILE_PATH = ROOT_DIR / "index.html"

# Initialize the API Router
router = APIRouter()

@router.get("/")
async def root_status():
    """
    Reads the index.html file content and serves it as HTMLResponse.
    """
    try:
        # Reads the content of the index.html file and returns it.
        html_content = STATUS_HTML_FILE_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        # Returns a simple HTML error message if the file is not found.
        # This helps check the file path during development.
        return HTMLResponse(
            content="<h1>Server is running, but index.html was not found.</h1><p>Check the path: " + str(HTML_FILE_PATH) + "</p>",
            status_code=500
        )

@router.post("/analyze", response_model=VibeResponse)
async def analyze(
        category: Category = Form(...),
        file: UploadFile = File(...)
):
    # 1. Process the Image and Extract User Context (Emotion, Age, Gender)
    image_bytes = await file.read()
    user_context = analyze_image_with_smart_ai(image_bytes)

    if not user_context:
        # If the vision pipeline fails to detect a face or extract data
        raise HTTPException(status_code=400, detail="Face could not be detected or analyzed.")

    # 2. Get Recommendations from the LLM (Gemini)
    recommendation_data = get_recommendations_from_gemini(user_context, category)

    if not recommendation_data:
        # If the LLM service or its retry mechanism fails
        raise HTTPException(status_code=500, detail="AI service failed to return a response.")

    # 3. Construct and Return the Final Response
    # Merge the user context (from Vision) with the recommendations (from LLM)
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