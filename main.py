from fastapi import FastAPI
from app.api.router import router

app = FastAPI(title="VibeLens API")

# Router'ı dahil et
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)