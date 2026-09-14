from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.image_routes import router as image_router

app = FastAPI(
    title="Truth Lens Backend API",
    description="Multimodal GenAI Authenticity & Image Deepfake Detection Service",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_router)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "app": "Truth Lens Backend API",
        "version": "2.0.0",
        "endpoints": ["/analyze/image"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
