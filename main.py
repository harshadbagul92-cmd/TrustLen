import os
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schemas import Evidence, ModalityEnum
from explainer import explain
from detectors.text import TextDetector
from detectors.audio import AudioDetector
from detectors.image import ImageDetector
from detectors.video import VideoDetector

app = FastAPI(
    title="Truth Lens API",
    description="GenAI classifier & explainer with non-confident verdict bands",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api.image_routes import router as image_router
app.include_router(image_router)

# Instantiate modular detectors
text_detector = TextDetector()
audio_detector = AudioDetector()
image_detector = ImageDetector()
video_detector = VideoDetector()


class TextAnalyzeRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None


@app.get("/")
def read_root():
    return {
        "status": "online",
        "app": "Truth Lens API",
        "supported_modalities": ["text", "image", "audio", "video"]
    }


@app.post("/analyze", response_model=Evidence)
async def analyze(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    api_key: Optional[str] = Form(None),
    provider: Optional[str] = Form(None)
):
    """
    Unified /analyze endpoint accepting text or media file upload.
    Auto-detects modality and routes to the appropriate detection lane.
    """
    if not text and not file:
        raise HTTPException(status_code=400, detail="Must provide either text string or media file upload.")

    evidence: Evidence = None
    meta = {}
    if api_key:
        meta["api_key"] = api_key
    if provider:
        meta["provider"] = provider

    try:
        # Handle text input
        if text and not file:
            evidence = text_detector.analyse(input_data=text, metadata=meta)

        # Handle media file input
        elif file:
            content_type = file.content_type or ""
            filename = file.filename or "uploaded_file"
            file_bytes = await file.read()
            file_meta = {
                "filename": filename,
                "content_type": content_type,
                "size_bytes": len(file_bytes),
                **meta
            }

            # Content-type or filename extension routing
            fn_lower = filename.lower()
            if content_type.startswith("image/") or any(fn_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]):
                evidence = image_detector.analyse(input_data=file_bytes, metadata=file_meta)

            elif content_type.startswith("audio/") or any(fn_lower.endswith(ext) for ext in [".mp3", ".wav", ".ogg", ".m4a", ".flac"]):
                evidence = audio_detector.analyse(input_data=file_bytes, metadata=file_meta)

            elif content_type.startswith("video/") or any(fn_lower.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]):
                evidence = video_detector.analyse(input_data=file_bytes, metadata=file_meta)

            elif content_type.startswith("text/") or fn_lower.endswith(".txt"):
                text_content = file_bytes.decode("utf-8", errors="ignore")
                evidence = text_detector.analyse(input_data=text_content, metadata=file_meta)

            else:
                # Default fallback routing based on content inspection or text
                try:
                    decoded = file_bytes.decode("utf-8")
                    evidence = text_detector.analyse(input_data=decoded, metadata=file_meta)
                except Exception:
                    # Default to image lane if binary un-routable
                    evidence = image_detector.analyse(input_data=file_bytes, metadata=file_meta)

        if not evidence:
            raise HTTPException(status_code=500, detail="Failed to generate evidence from input.")

        # Generate plain-English grounded explanation
        evidence.explanation = explain(evidence=evidence, api_key=api_key, provider=provider or "openai")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")

    return evidence


@app.post("/analyze/json", response_model=Evidence)
def analyze_json(payload: TextAnalyzeRequest):
    """Convenience endpoint for pure JSON text analysis."""
    try:
        evidence = text_detector.analyse(input_data=payload.text, metadata=payload.metadata)
        api_key = payload.metadata.get("api_key") if payload.metadata else None
        provider = payload.metadata.get("provider", "openai") if payload.metadata else "openai"
        evidence.explanation = explain(evidence=evidence, api_key=api_key, provider=provider)
        return evidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON analysis error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
