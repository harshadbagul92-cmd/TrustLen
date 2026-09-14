import os
from typing import Dict
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


class Settings:
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    HF_IMAGE_MODEL: str = os.getenv("HF_IMAGE_MODEL", "umm-maybe/AI-image-detector")
    HF_PROVIDER: str = os.getenv("HF_PROVIDER", "hf-inference")
    IMAGE_DETECTOR_MODE: str = os.getenv("IMAGE_DETECTOR_MODE", "mock").lower()

    # Thresholds
    CONFIDENCE_THRESHOLD_UNCERTAIN: float = float(os.getenv("CONFIDENCE_THRESHOLD_UNCERTAIN", "0.60"))
    CONFIDENCE_THRESHOLD_MODERATE: float = float(os.getenv("CONFIDENCE_THRESHOLD_MODERATE", "0.80"))

    # Configurable Label Mapping for Verified Hugging Face Models
    # Converts exact model output labels -> LIKELY_SYNTHETIC | LIKELY_AUTHENTIC | UNCERTAIN
    IMAGE_MODEL_LABEL_MAP: Dict[str, str] = {
        # Primary Model: umm-maybe/AI-image-detector & Organika/sdxl-detector
        "artificial": "LIKELY_SYNTHETIC",
        "human": "LIKELY_AUTHENTIC",

        # Backup Model: prithivMLmods/Deep-Fake-Detector-v2-Model
        "deepfake": "LIKELY_SYNTHETIC",
        "realism": "LIKELY_AUTHENTIC",

        # Candidate Model 2: dima806/ai_vs_real_image_detection
        "fake": "LIKELY_SYNTHETIC",
        "real": "LIKELY_AUTHENTIC"
    }


settings = Settings()
