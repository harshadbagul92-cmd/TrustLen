import os
import io
import logging
from typing import Dict, Any, Optional
from PIL import Image

from backend.config import settings
from backend.schemas.detector_result import PredictionEnum

logger = logging.getLogger(__name__)


class DetectorOutput:
    def __init__(
        self,
        prediction: PredictionEnum,
        raw_score: float,
        raw_label: str,
        provider: str,
        model_name: str,
        raw_labels: Optional[list] = None,
        raw_scores: Optional[list] = None,
        task: str = "whole_image_synthesis",
        signal: str = "unknown",
        reliability: str = "medium",
        limitations: Optional[list] = None,
        warning: Optional[str] = None
    ):
        self.prediction = prediction
        self.raw_score = raw_score
        self.raw_label = raw_label
        self.raw_labels = raw_labels or [raw_label] if raw_label else []
        self.raw_scores = raw_scores or [raw_score]
        self.provider = provider
        self.model_name = model_name
        self.task = task
        self.signal = signal if signal != "unknown" else ("synthetic" if prediction == PredictionEnum.LIKELY_SYNTHETIC else ("authentic" if prediction == PredictionEnum.LIKELY_AUTHENTIC else "unknown"))
        self.reliability = reliability
        self.limitations = limitations or [
            "umm-maybe/AI-image-detector is a global whole-image generator detector, not a localized facial deepfake/swap classifier."
        ]
        self.warning = warning

    def to_visual_output(self):
        from backend.schemas.detector_result import VisualDetectorOutput
        score_dict = dict(zip(self.raw_labels, self.raw_scores)) if self.raw_labels and self.raw_scores else {self.raw_label: self.raw_score}
        return VisualDetectorOutput(
            model=self.model_name,
            task=self.task,
            raw_label=self.raw_label,
            scores=score_dict,
            detector_score=round(self.raw_score, 4),
            signal=self.signal,
            reliability=self.reliability,
            limitations=self.limitations
        )


class HuggingFaceImageDetector:
    """
    Hugging Face Inference API Adapter for AI-generated image detection.
    Encapsulates model requests, rate limit/timeout handling, label normalization,
    and deterministic mock mode fallback (used ONLY when explicitly configured).
    """

    def __init__(self):
        self.token = settings.HF_TOKEN
        self.model_name = settings.HF_IMAGE_MODEL
        self.provider = settings.HF_PROVIDER
        self.mode = settings.IMAGE_DETECTOR_MODE
        self.label_map = settings.IMAGE_MODEL_LABEL_MAP

    def normalize_label(self, raw_label: str) -> PredictionEnum:
        """Converts raw model output string to normalized PredictionEnum."""
        clean_label = str(raw_label).strip().lower()
        mapped = self.label_map.get(clean_label)

        if mapped == "LIKELY_SYNTHETIC":
            return PredictionEnum.LIKELY_SYNTHETIC
        elif mapped == "LIKELY_AUTHENTIC":
            return PredictionEnum.LIKELY_AUTHENTIC
        
        # General fallbacks for unmapped labels
        if any(k in clean_label for k in ["artificial", "ai", "synthetic", "fake", "deepfake"]):
            return PredictionEnum.LIKELY_SYNTHETIC
        elif any(k in clean_label for k in ["human", "real", "authentic"]):
            return PredictionEnum.LIKELY_AUTHENTIC
        
        return PredictionEnum.UNCERTAIN

    def detect_mock(self, image: Image.Image, filename: str = "") -> DetectorOutput:
        """Deterministic mock detector for development and offline testing."""
        fn_lower = filename.lower()

        if any(k in fn_lower for k in ["c2pa", "midjourney", "fake", "swap", "sora", "deepfake", "stablediffusion", "dalle", "photoshop", "edit"]):
            return DetectorOutput(
                prediction=PredictionEnum.LIKELY_SYNTHETIC,
                raw_score=0.94,
                raw_label="artificial",
                raw_labels=["artificial", "human"],
                raw_scores=[0.94, 0.06],
                provider="mock-inference",
                model_name=f"{self.model_name} (Mock Mode)"
            )
        elif any(k in fn_lower for k in ["compressed", "whatsapp", "lowfps"]):
            return DetectorOutput(
                prediction=PredictionEnum.UNCERTAIN,
                raw_score=0.52,
                raw_label="uncertain_quality",
                raw_labels=["artificial", "human"],
                raw_scores=[0.52, 0.48],
                provider="mock-inference",
                model_name=f"{self.model_name} (Mock Mode)",
                warning="Image quality / compression prevents definitive model score."
            )
        elif any(k in fn_lower for k in ["authentic", "camera", "real", "safe", "nature", "portrait", "street", "landscape", "family"]):
            return DetectorOutput(
                prediction=PredictionEnum.LIKELY_AUTHENTIC,
                raw_score=0.88,
                raw_label="human",
                raw_labels=["human", "artificial"],
                raw_scores=[0.88, 0.12],
                provider="mock-inference",
                model_name=f"{self.model_name} (Mock Mode)"
            )

        # Default fallback score based on basic image properties
        w, h = image.size
        is_square = abs(w - h) < 5
        raw_score = 0.85 if (is_square and w in [512, 1024]) else 0.82
        pred = PredictionEnum.LIKELY_SYNTHETIC if (is_square and w in [512, 1024]) else PredictionEnum.LIKELY_AUTHENTIC
        top_lbl = "artificial" if pred == PredictionEnum.LIKELY_SYNTHETIC else "human"
        other_lbl = "human" if top_lbl == "artificial" else "artificial"

        return DetectorOutput(
            prediction=pred,
            raw_score=raw_score,
            raw_label=top_lbl,
            raw_labels=[top_lbl, other_lbl],
            raw_scores=[raw_score, round(1.0 - raw_score, 4)],
            provider="mock-inference",
            model_name=f"{self.model_name} (Mock Mode)"
        )

    def detect(self, image: Image.Image, filename: str = "") -> DetectorOutput:
        """
        Executes AI image detection via Hugging Face Inference API or Mock Mode.
        Enforces strict mode: when mode == 'api', never silently fallback to mock mode.
        """
        # Step 1: Use mock mode ONLY if mode is explicitly 'mock'
        if self.mode == "mock":
            logger.info("[HF Detector] Using explicit mock mode.")
            return self.detect_mock(image, filename=filename)

        # Step 2: Ensure token exists when in API mode
        if not self.token:
            logger.warning("[HF Detector] IMAGE_DETECTOR_MODE=api but HF_TOKEN is missing.")
            return DetectorOutput(
                prediction=PredictionEnum.UNCERTAIN,
                raw_score=0.50,
                raw_label="no_token",
                raw_labels=["no_token"],
                raw_scores=[0.50],
                provider=self.provider,
                model_name=self.model_name,
                warning="HF_TOKEN missing in environment. Cannot perform real API inference."
            )

        # Step 3: Call Hugging Face Inference API using huggingface_hub.InferenceClient
        try:
            import tempfile
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=self.token, model=self.model_name)

            # Save PIL image to a temporary file path so InferenceClient attaches Content-Type header
            fmt = image.format if image.format and image.format in ["JPEG", "PNG", "WEBP"] else "JPEG"
            ext = f".{fmt.lower()}"
            if ext == ".jpeg":
                ext = ".jpg"

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                temp_path = tmp.name
                image.save(temp_path, format=fmt)

            try:
                res = client.image_classification(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            if not res or not isinstance(res, list):
                return DetectorOutput(
                    prediction=PredictionEnum.UNCERTAIN,
                    raw_score=0.50,
                    raw_label="invalid_response",
                    raw_labels=["invalid_response"],
                    raw_scores=[0.50],
                    provider=self.provider,
                    model_name=self.model_name,
                    warning="Hugging Face API returned empty or malformed output."
                )

            raw_labels = []
            raw_scores = []
            for item in res:
                lbl = getattr(item, "label", None) or (item.get("label") if isinstance(item, dict) else None)
                scr = getattr(item, "score", None) or (item.get("score") if isinstance(item, dict) else 0.0)
                if lbl is not None:
                    raw_labels.append(str(lbl))
                    raw_scores.append(float(scr))

            if not raw_labels or not raw_scores:
                return DetectorOutput(
                    prediction=PredictionEnum.UNCERTAIN,
                    raw_score=0.50,
                    raw_label="empty_labels",
                    raw_labels=["empty_labels"],
                    raw_scores=[0.50],
                    provider=self.provider,
                    model_name=self.model_name,
                    warning="No classification labels returned from Hugging Face model."
                )

            top_raw_label = raw_labels[0]
            top_raw_score = raw_scores[0]
            prediction = self.normalize_label(top_raw_label)

            return DetectorOutput(
                prediction=prediction,
                raw_score=top_raw_score,
                raw_label=top_raw_label,
                raw_labels=raw_labels,
                raw_scores=raw_scores,
                provider=self.provider,
                model_name=self.model_name
            )

        except Exception as e:
            err_msg = str(e)
            if self.token and self.token in err_msg:
                err_msg = err_msg.replace(self.token, "[TOKEN_REDACTED]")
            
            logger.error(f"[HF Detector API Error] {err_msg}")
            
            # Return UNCERTAIN with warning in strict API mode (NO silent mock fallback)
            return DetectorOutput(
                prediction=PredictionEnum.UNCERTAIN,
                raw_score=0.50,
                raw_label="api_error",
                raw_labels=["api_error"],
                raw_scores=[0.50],
                provider=self.provider,
                model_name=self.model_name,
                warning=f"Hugging Face API inference failed: {err_msg}"
            )

