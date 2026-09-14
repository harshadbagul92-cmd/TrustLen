# Truth Lens — Phase 2: Multimodal GenAI Authenticity Checker

Truth Lens is a GenAI authenticity detection platform providing evidence-grounded image, audio, video, and text analysis without returning binary "real vs fake" claims.

---

## Phase 2 Architecture: Image Authenticity Module

```
IMAGE UPLOAD
     ↓
IMAGE VALIDATION / PREPROCESSING
     ↓
PROVENANCE / METADATA CHECK
     ↓
FACE / REGION DETECTION
     ↓
AI IMAGE DETECTOR (Hugging Face Adapter)
     ↓
EVIDENCE ENGINE
     ↓
CONFIDENCE ENGINE
     ↓
STRUCTURED DETECTOR RESULT
     ↓
LLM EXPLAINER
```

---

## Project Structure

```
backend/
├── api/
│   └── image_routes.py         # POST /analyze/image
├── detectors/
│   └── image/
│       ├── __init__.py
│       ├── image_detector.py
│       ├── hf_image_detector.py # HF Inference Client & Mock Adapter
│       ├── preprocessing.py     # Image PIL validation & EXIF reader
│       ├── provenance.py        # EXIF & C2PA manifest extractor
│       ├── face_detection.py    # Region/face bounding box detector
│       └── evidence.py          # Evidence synthesis engine
├── schemas/
│   └── detector_result.py       # Pydantic v2 DetectorResult schema
├── services/
│   ├── confidence.py            # Confidence engine (low, moderate, high)
│   └── image_analysis.py       # Pipeline orchestrator
├── config.py                    # Environment & threshold configuration
└── main.py                      # FastAPI entry point
```

---

## Environment Configuration

Copy `.env.example` to `.env` and fill in optional Hugging Face API tokens:

```ini
HF_TOKEN=your_huggingface_token_here
HF_IMAGE_MODEL=umm-maybe/AI-image-detector
HF_PROVIDER=hf-inference
IMAGE_DETECTOR_MODE=mock
```

- `IMAGE_DETECTOR_MODE=mock`: Returns instant deterministic responses for testing without consuming Hugging Face API credits.
- `IMAGE_DETECTOR_MODE=api`: Calls the Hugging Face Serverless Inference API via `huggingface_hub.InferenceClient`.

---

## API Endpoints

### 1. Image Authenticity Endpoint (`POST /analyze/image`)

**Request:**
- File upload (`multipart/form-data`) under parameter `file` (JPG, JPEG, PNG, WEBP max 20MB).

**Response (`APIResponse` JSON):**
```json
{
  "success": true,
  "result": {
    "modality": "image",
    "prediction": "likely_synthetic",
    "confidence": 0.87,
    "confidence_level": "high",
    "uncertainty": 0.13,
    "evidence": [
      {
        "category": "ai_detector_signal",
        "description": "The image detector produced a high synthetic score (0.94) with label 'artificial'.",
        "score": 0.94,
        "location": null
      }
    ],
    "provenance": {
      "metadata_found": true,
      "c2pa_found": true,
      "editing_software": "Midjourney v6.0"
    },
    "regions": [],
    "warnings": [
      "AI-generated image detection is probabilistic and may fail on unseen generators."
    ],
    "model": {
      "provider": "hf-inference",
      "model": "umm-maybe/AI-image-detector"
    }
  }
}
```

---

## Running Automated Tests

Run the complete test suite (41 unit tests):

```powershell
python -m unittest discover -s tests
```

---

## Launching the Web UI & API Server

### 1. Launch FastAPI Server
```powershell
uvicorn main:app --reload --port 8000
```

### 2. Launch Streamlit Frontend
```powershell
streamlit run app.py
```