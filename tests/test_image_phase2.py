import os
import sys
import io
import unittest
import warnings
from unittest.mock import patch, MagicMock
from PIL import Image

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main import app
from backend.services.image_analysis import ImageAnalysisService
from backend.detectors.image.hf_image_detector import HuggingFaceImageDetector, DetectorOutput
from backend.schemas.detector_result import PredictionEnum, ConfidenceLevelEnum


def create_test_image_bytes(fmt: str = "JPEG", size: tuple = (200, 200), color: str = "blue") -> bytes:
    """Helper to generate valid dummy PIL image bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class TestPhase2ImageAuthenticity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.service = ImageAnalysisService()
        cls.detector = HuggingFaceImageDetector()

    # 1. Valid JPG
    def test_1_valid_jpg_upload(self):
        jpg_bytes = create_test_image_bytes(fmt="JPEG")
        files = {"file": ("test_photo.jpg", jpg_bytes, "image/jpeg")}
        response = self.client.post("/analyze/image", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["result"]["modality"], "image")
        self.assertIn("prediction", data["result"])

    # 2. Valid PNG
    def test_2_valid_png_upload(self):
        png_bytes = create_test_image_bytes(fmt="PNG")
        files = {"file": ("test_graphic.png", png_bytes, "image/png")}
        response = self.client.post("/analyze/image", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])

    # 3. Invalid File Extension / Unsupported File
    def test_3_invalid_file_type(self):
        files = {"file": ("document.pdf", b"%PDF-1.4 dummy content", "application/pdf")}
        response = self.client.post("/analyze/image", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file extension", response.json()["detail"])

    # 4. Corrupted Image Bytes
    def test_4_corrupted_image(self):
        files = {"file": ("corrupt.jpg", b"NOT_AN_IMAGE_DATA_STREAM", "image/jpeg")}
        response = self.client.post("/analyze/image", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Corrupted or undecodable image", response.json()["detail"])

    # 5. Missing HF Token Handling (Falls back gracefully to Mock Mode)
    def test_5_missing_hf_token(self):
        img = Image.new("RGB", (100, 100))
        with patch.object(self.detector, 'token', ''):
            output = self.detector.detect(img, filename="sample.jpg")
            self.assertIsNotNone(output)
            self.assertIn("Mock Mode", output.model_name)

    # 6. Missing Model Handling
    def test_6_missing_model_handling(self):
        img = Image.new("RGB", (100, 100))
        with patch.object(self.detector, 'model_name', ''):
            output = self.detector.detect(img, filename="sample.jpg")
            self.assertIsNotNone(output)

    # 7. Mocked Hugging Face Response
    @patch('huggingface_hub.InferenceClient.image_classification')
    def test_7_mocked_hf_response(self, mock_hf):
        mock_hf.return_value = [{"label": "artificial", "score": 0.96}]

        img = Image.new("RGB", (100, 100))
        detector = HuggingFaceImageDetector()
        detector.mode = "api"
        detector.token = "hf_dummy_token"

        output = detector.detect(img, filename="ai_generated.jpg")
        self.assertEqual(output.prediction, PredictionEnum.LIKELY_SYNTHETIC)
        self.assertEqual(output.raw_score, 0.96)

    # 8. Hugging Face Timeout / Failure Handling
    @patch('huggingface_hub.InferenceClient.image_classification')
    def test_8_hf_timeout_failure_fallback(self, mock_hf):
        mock_hf.side_effect = TimeoutError("HuggingFace API Connection Timed Out")

        img = Image.new("RGB", (100, 100))
        detector = HuggingFaceImageDetector()
        detector.mode = "api"
        detector.token = "hf_dummy_token"

        output = detector.detect(img, filename="timeout_test.jpg")
        self.assertIsNotNone(output)
        self.assertIn("Timed Out", output.warning)

    # 9. Unknown Model Label Mapping Fallback
    def test_9_unknown_model_label(self):
        detector = HuggingFaceImageDetector()
        pred = detector.normalize_label("unknown_custom_model_label_xyz")
        self.assertEqual(pred, PredictionEnum.UNCERTAIN)

    # 10. Successful Full Image Analysis Flow
    def test_10_full_image_analysis_flow(self):
        jpg_bytes = create_test_image_bytes(fmt="JPEG", size=(300, 300))
        result = self.service.analyze_image(jpg_bytes, filename="c2pa_midjourney_photo.jpg")

        self.assertEqual(result.modality, "image")
        self.assertIn(result.prediction, [PredictionEnum.LIKELY_SYNTHETIC, PredictionEnum.LIKELY_AUTHENTIC, PredictionEnum.UNCERTAIN])
        self.assertIn(result.confidence_level, [ConfidenceLevelEnum.LOW, ConfidenceLevelEnum.MODERATE, ConfidenceLevelEnum.HIGH])
        self.assertGreater(len(result.evidence), 0)
        self.assertIsNotNone(result.provenance)
        self.assertIsNotNone(result.model)


if __name__ == "__main__":
    unittest.main()
