"""
TrustLens HTTP API.

    uvicorn api:app --reload
    http://127.0.0.1:8000/docs

Nothing submitted here is written to disk or kept in memory after the
response is returned. The single exception is video, which OpenCV can only
read from a path; that goes through trustlens.privacy.scratch_file, which
deletes the file in a finally block. /privacy reports the live counters.
"""
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trustlens import __version__, config, detectors, llm, privacy
from trustlens.explainer import explain
from trustlens.schemas import Evidence

@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    privacy.purge()          # leave nothing behind on shutdown


app = FastAPI(
    lifespan=lifespan,
    title="TrustLens API",
    version=__version__,
    description="Classifier and explainer for scam text, manipulated images, "
                "cloned voices and deepfake video. Never returns a binary "
                "real/fake verdict; uncertainty is a first-class answer.",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["POST", "GET"], allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Web interface
# ---------------------------------------------------------------------------
# The React app is served by this same process, so there is ONE command to run
# at demo time. Its libraries are vendored under web/vendor rather than pulled
# from a CDN, so the interface still works if the venue wifi does not.
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        path = os.path.join(WEB_DIR, "favicon.svg")
        return FileResponse(path) if os.path.exists(path) else FileResponse(
            os.path.join(WEB_DIR, "index.html"))


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000)
    explain_with_model: bool = True


@app.get("/health")
def health():
    model_ok, model_reason = llm.available()
    return {
        "status": "ok",
        "version": __version__,
        "language_model": {"available": model_ok, "reason": model_reason},
        "audio_video_decoder": detectors_media_ready(),
        "verdict_bands": {
            "likely_authentic": f"< {config.AUTHENTIC_BELOW}",
            "uncertain": f"{config.AUTHENTIC_BELOW} - {config.MANIPULATED_ABOVE}",
            "likely_manipulated": f"> {config.MANIPULATED_ABOVE}",
        },
    }


def detectors_media_ready() -> bool:
    from trustlens.detectors import media
    return media.ffmpeg_available()


@app.get("/privacy")
def privacy_status():
    """Live proof that nothing is being retained."""
    stats = privacy.stats()
    return {
        "stores_uploads": False,
        "stores_results": False,
        "retention": "none - every analysis is in-memory and discarded with the response",
        "scratch_files": stats,
    }


@app.post("/analyze/text", response_model=Evidence)
def analyze_text(request: TextRequest):
    evidence = detectors.get(detectors.Modality.TEXT).analyse(request.text, {})
    evidence.explanation = explain(evidence, use_model=request.explain_with_model)
    return evidence


@app.post("/analyze", response_model=Evidence)
async def analyze(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    explain_with_model: bool = Form(True),
):
    """
    One endpoint for everything. Send `text`, or upload a `file`; the modality
    is detected from the content type and extension, never from the rest of
    the filename.
    """
    if not file and not (text or "").strip():
        raise HTTPException(400, "Provide either a text field or a file upload.")

    if text and not file:
        evidence = detectors.get(detectors.Modality.TEXT).analyse(text.strip(), {})
    else:
        payload = await file.read()
        if not payload:
            raise HTTPException(400, "The uploaded file was empty.")
        if len(payload) > config.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File exceeds the {config.MAX_UPLOAD_MB} MB limit.")
        evidence = detectors.analyse(payload, file.filename or "upload",
                                     file.content_type or "")
        del payload                      # drop the bytes as soon as we are done

    evidence.explanation = explain(evidence, use_model=explain_with_model)
    return evidence


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
