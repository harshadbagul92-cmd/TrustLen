"""
Trained CLIP (ViT-Large/14) AI-image detection with Platt scaling.

Loads openai/clip-vit-large-patch14 and the calibrated classifier from
models/image_clf_calibrated.joblib once on module load.
Extracts L2-normalized image embeddings matching the training pipeline
in scripts/train_clip_baseline.py and computes calibrated probabilities.
"""
import os
import warnings
from typing import Optional

from PIL import Image

from .. import config
from ..schemas import Direction, Signal

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "openai/clip-vit-large-patch14"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CALIBRATED_MODEL_PATH = os.path.join(REPO_ROOT, "models", "image_clf_calibrated.joblib")
BASE_MODEL_PATH = os.path.join(REPO_ROOT, "models", "image_clf.joblib")

# Consistent with image lane calibration:
# Single noise anomaly = 0.55, VLM = 0.60-0.80, Corroborated splice = 0.85
CLIP_AI_WEIGHT = 0.75

_MODEL_LOADED: bool = False
_LOAD_ERROR: Optional[str] = None
_clip_model = None
_clip_processor = None
_classifier = None
_device: str = "cpu"


def _initialize_model():
    """
    Load CLIP processor, model, and calibrated classifier once.
    Fails safely without raising so the lane can abstain gracefully.
    """
    global _MODEL_LOADED, _LOAD_ERROR, _clip_model, _clip_processor, _classifier, _device

    # Offline test flag guard
    if os.environ.get("TRUSTLENS_NO_MODEL", "").lower() in ("1", "true", "yes") or config.NO_MODEL:
        _MODEL_LOADED = False
        _LOAD_ERROR = "disabled_by_TRUSTLENS_NO_MODEL"
        return

    # Check classifier existence (prefer calibrated, fallback to uncalibrated)
    clf_path = CALIBRATED_MODEL_PATH if os.path.isfile(CALIBRATED_MODEL_PATH) else BASE_MODEL_PATH
    if not os.path.isfile(clf_path):
        _MODEL_LOADED = False
        _LOAD_ERROR = f"Classifier model file not found at {CALIBRATED_MODEL_PATH}"
        return

    try:
        import joblib
        import torch
        from transformers import CLIPModel, CLIPProcessor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(_device)
        _clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _clip_model.eval()

        _classifier = joblib.load(clf_path)
        _MODEL_LOADED = True
        _LOAD_ERROR = None
    except Exception as exc:
        _MODEL_LOADED = False
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"


# Run once on module load
_initialize_model()


def analyse_ai_generation(image: Image.Image) -> Signal:
    """
    Assess whether an image is AI-generated using calibrated CLIP ViT-Large/14 features.

    Returns:
      Signal with name 'clip_ai_classifier'.
      - Direction is MANIPULATION (>= 0.50) or AUTHENTIC (< 0.50) for labeling.
      - Value is the raw calibrated probability of AI (0.0 to 1.0).
      - Weight is consistent with image lane calibration (0.75).
      - If unavailable or offline, returns Direction.LIMITATION.
    """
    # 1. Offline mode or model unavailable
    if os.environ.get("TRUSTLENS_NO_MODEL", "").lower() in ("1", "true", "yes") or config.NO_MODEL:
        return Signal(
            name="clip_ai_classifier",
            direction=Direction.LIMITATION,
            human="A trained image classifier is unavailable in offline mode (TRUSTLENS_NO_MODEL=1).",
            value="disabled_by_TRUSTLENS_NO_MODEL",
            weight=0.20,
            where="whole image",
        )

    if not _MODEL_LOADED or _classifier is None or _clip_model is None or _clip_processor is None:
        reason = _LOAD_ERROR or "model_uninitialized"
        return Signal(
            name="clip_ai_classifier",
            direction=Direction.LIMITATION,
            human=f"A trained image classifier could not be loaded ({reason}).",
            value=reason,
            weight=0.20,
            where="whole image",
        )

    # 2. Extract embedding matching train_clip_baseline.py
    try:
        import torch

        rgb_image = image.convert("RGB")
        inputs = _clip_processor(images=rgb_image, return_tensors="pt").to(_device)

        with torch.no_grad():
            feat = _clip_model.get_image_features(**inputs)
            if hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            elif isinstance(feat, (tuple, list)):
                feat = feat[0]
            # L2 normalization
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embedding = feat.cpu().numpy()

        # 3. Predict probability of class 1 (AI)
        probs = _classifier.predict_proba(embedding)[0]
        prob_ai = float(probs[1])
        prob_ai = max(0.0, min(1.0, prob_ai))
        pct = round(prob_ai * 100)

        # Label direction for UI interpretation
        if prob_ai >= 0.50:
            direction = Direction.MANIPULATION
            human = f"A trained image classifier estimates a {pct}% probability this image is AI-generated."
        else:
            direction = Direction.AUTHENTIC
            human = (f"A trained image classifier estimates a {pct}% probability this image is AI-generated "
                     f"(no strong generation patterns detected).")

        return Signal(
            name="clip_ai_classifier",
            direction=direction,
            human=human,
            value=round(prob_ai, 4),
            weight=CLIP_AI_WEIGHT,
            where="whole image",
        )

    except Exception as exc:
        return Signal(
            name="clip_ai_classifier",
            direction=Direction.LIMITATION,
            human=f"Trained image classifier inference failed ({type(exc).__name__}: {exc}).",
            value=str(exc),
            weight=0.20,
            where="whole image",
        )
