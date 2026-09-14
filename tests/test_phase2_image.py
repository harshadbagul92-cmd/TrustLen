import io
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

from backend.detectors.image.preprocessing import ImagePreprocessor
from backend.detectors.image.provenance import ProvenanceChecker
from backend.detectors.image.hf_image_detector import HuggingFaceImageDetector, DetectorOutput
from backend.detectors.image.evidence import EvidenceEngine
from backend.services.confidence import ConfidenceEngine
from backend.services.image_analysis import ImageAnalysisService
from backend.schemas.detector_result import (
    PredictionEnum, ConfidenceLevelEnum, DetectorResult, EvidenceItem, EvidenceCategoryEnum, ModelMeta
)


def create_sample_jpeg_bytes(width=100, height=100, color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def create_sample_png_bytes(width=100, height=100) -> bytes:
    img = Image.new("RGB", (width, height), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestPhase2ImageModule(unittest.TestCase):

    # 1. Preprocessing Tests
    def test_preprocessor_valid_jpg(self):
        preprocessor = ImagePreprocessor()
        data = create_sample_jpeg_bytes(200, 200)
        res = preprocessor.preprocess(data, filename="sample.jpg")
        self.assertEqual(res.width, 200)
        self.assertEqual(res.height, 200)
        self.assertEqual(res.format_name, "JPEG")
        self.assertEqual(len(res.sha256), 64)

    def test_preprocessor_valid_png(self):
        preprocessor = ImagePreprocessor()
        data = create_sample_png_bytes(150, 150)
        res = preprocessor.preprocess(data, filename="sample.png")
        self.assertEqual(res.width, 150)
        self.assertEqual(res.height, 150)
        self.assertEqual(res.format_name, "PNG")

    def test_preprocessor_corrupted_image(self):
        preprocessor = ImagePreprocessor()
        with self.assertRaises(ValueError):
            preprocessor.preprocess(b"NOT_AN_IMAGE_DATA_CORRUPTED", filename="test.jpg")

    def test_preprocessor_empty_bytes(self):
        preprocessor = ImagePreprocessor()
        with self.assertRaises(ValueError):
            preprocessor.preprocess(b"", filename="test.jpg")

    # 2. Provenance Tests
    def test_provenance_missing_exif(self):
        checker = ProvenanceChecker()
        data = create_sample_jpeg_bytes()
        prov, evidence = checker.check(data, exif_tags={}, filename="sample.jpg")
        self.assertFalse(prov.metadata_found)
        self.assertFalse(prov.c2pa_found)
        self.assertTrue(any("No digital provenance" in e.description for e in evidence))

    def test_provenance_with_exif_software(self):
        checker = ProvenanceChecker()
        data = create_sample_jpeg_bytes()
        exif = {"Software": "Adobe Photoshop 2024", "Make": "Canon", "Model": "EOS R5"}
        prov, evidence = checker.check(data, exif_tags=exif, filename="sample.jpg")
        self.assertTrue(prov.metadata_found)
        self.assertEqual(prov.camera_model, "Canon EOS R5")
        self.assertEqual(prov.editing_software, "Adobe Photoshop 2024")
        self.assertTrue(any("Photoshop" in e.description for e in evidence))

    # 3. Detector API Tests (Mocking HF Client)
    def test_detector_normalize_labels(self):
        detector = HuggingFaceImageDetector()
        self.assertEqual(detector.normalize_label("artificial"), PredictionEnum.LIKELY_SYNTHETIC)
        self.assertEqual(detector.normalize_label("human"), PredictionEnum.LIKELY_AUTHENTIC)
        self.assertEqual(detector.normalize_label("ai_generated"), PredictionEnum.LIKELY_SYNTHETIC)
        self.assertEqual(detector.normalize_label("real"), PredictionEnum.LIKELY_AUTHENTIC)
        self.assertEqual(detector.normalize_label("unknown_xyz"), PredictionEnum.UNCERTAIN)

    @patch("huggingface_hub.InferenceClient")
    def test_detector_api_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.image_classification.return_value = [
            {"label": "artificial", "score": 0.94},
            {"label": "human", "score": 0.06}
        ]
        mock_client_cls.return_value = mock_client

        detector = HuggingFaceImageDetector()
        detector.mode = "api"
        detector.token = "hf_test_token"

        img = Image.new("RGB", (100, 100))
        out = detector.detect(img, filename="test.jpg")

        self.assertEqual(out.prediction, PredictionEnum.LIKELY_SYNTHETIC)
        self.assertEqual(out.raw_score, 0.94)
        self.assertEqual(out.raw_label, "artificial")
        self.assertEqual(out.raw_labels, ["artificial", "human"])
        self.assertEqual(out.raw_scores, [0.94, 0.06])

    @patch("huggingface_hub.InferenceClient")
    def test_detector_api_failure_no_fallback(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.image_classification.side_effect = Exception("503 Service Unavailable")
        mock_client_cls.return_value = mock_client

        detector = HuggingFaceImageDetector()
        detector.mode = "api"
        detector.token = "hf_test_token"

        img = Image.new("RGB", (100, 100))
        out = detector.detect(img, filename="test.jpg")

        # Strict API mode returns UNCERTAIN with error warning
        self.assertEqual(out.prediction, PredictionEnum.UNCERTAIN)
        self.assertIsNotNone(out.warning)
        self.assertIn("503 Service Unavailable", out.warning)

    # 4. Confidence Engine Tests
    def test_confidence_engine_thresholds(self):
        engine = ConfidenceEngine()

        # High score >= 0.80
        pred, conf, level, unc, reason, *rest = engine.evaluate(
            raw_prediction=PredictionEnum.LIKELY_SYNTHETIC,
            raw_score=0.88,
            provenance=MagicMock(c2pa_found=False, c2pa_ai_declared=False, c2pa_present=False),
            evidence=[]
        )
        self.assertEqual(pred, PredictionEnum.LIKELY_SYNTHETIC)
        self.assertEqual(level, ConfidenceLevelEnum.HIGH)

        # Moderate score 0.60 - 0.80
        pred, conf, level, unc, reason, *rest = engine.evaluate(
            raw_prediction=PredictionEnum.LIKELY_AUTHENTIC,
            raw_score=0.72,
            provenance=MagicMock(c2pa_found=False, c2pa_ai_declared=False, c2pa_present=False),
            evidence=[]
        )
        self.assertEqual(pred, PredictionEnum.LIKELY_AUTHENTIC)
        self.assertEqual(level, ConfidenceLevelEnum.MODERATE)

        # Low score < 0.60 -> UNCERTAIN
        pred, conf, level, unc, reason, *rest = engine.evaluate(
            raw_prediction=PredictionEnum.LIKELY_SYNTHETIC,
            raw_score=0.45,
            provenance=MagicMock(c2pa_found=False, c2pa_ai_declared=False, c2pa_present=False),
            evidence=[]
        )
        self.assertEqual(pred, PredictionEnum.UNCERTAIN)
        self.assertEqual(level, ConfidenceLevelEnum.LOW)

    # 5. Full Service Integration Test
    def test_image_analysis_service_flow(self):
        service = ImageAnalysisService()
        # Mock detector to return authentic result
        service.hf_detector.detect = MagicMock(return_value=DetectorOutput(
            prediction=PredictionEnum.LIKELY_AUTHENTIC,
            raw_score=0.85,
            raw_label="human",
            raw_labels=["human", "artificial"],
            raw_scores=[0.85, 0.15],
            provider="hf-inference",
            model_name="umm-maybe/AI-image-detector"
        ))

        # Use an uncompressed test image file > 15 KB
        real_img_path = r"c:\Users\Harshad Bagul\OneDrive\Desktop\ravet pccoe\tests\data\images\authentic_nature_1.jpg"
        with open(real_img_path, "rb") as f:
            data = f.read()

        result = service.analyze_image(data, filename="authentic_nature_1.jpg")

        self.assertIsInstance(result, DetectorResult)
        self.assertEqual(result.modality, "image")
        self.assertEqual(result.prediction, PredictionEnum.LIKELY_AUTHENTIC)
        self.assertEqual(result.detector_score, 0.85)
        self.assertEqual(result.model.name, "umm-maybe/AI-image-detector")
        self.assertNotEqual(result.explanation, "")


if __name__ == "__main__":
    unittest.main()
