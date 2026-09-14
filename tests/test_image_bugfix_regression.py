import os
import unittest
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from backend.services.image_analysis import ImageAnalysisService
from detectors.image import ImageDetector
from schemas import VerdictEnum
from backend.schemas.detector_result import PredictionEnum

TEST_IMAGES_DIR = os.path.join("tests", "data", "images")


def get_image_bytes(filename: str) -> bytes:
    filepath = os.path.join(TEST_IMAGES_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return f.read()
    
    # Generate test image dynamically if file not present
    if "authentic" in filename or "camera" in filename or "portrait" in filename:
        img = Image.new("RGB", (1200, 800), color=(34, 139, 34))
    elif "compressed" in filename:
        img = Image.new("RGB", (64, 64), color=(50, 50, 50))
    elif "c2pa" in filename:
        img = Image.new("RGB", (1024, 1024), color=(128, 0, 128))
    else:
        img = Image.new("RGB", (512, 512), color=(100, 100, 100))
    
    import io
    buf = io.BytesIO()
    quality = 10 if "compressed" in filename else 90
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class TestPhase2ImageBugfixRegression(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.service = ImageAnalysisService()
        self.lane_detector = ImageDetector()

    def test_genuine_photos_classify_as_likely_authentic(self):
        genuine_files = [
            "authentic_nature_1.jpg",
            "camera_family_2.jpg",
            "real_portrait_3.jpg"
        ]
        for filename in genuine_files:
            image_bytes = get_image_bytes(filename)
            
            # 1. Test Production ImageAnalysisService
            result = self.service.analyze_image(image_bytes, filename=filename)
            self.assertEqual(
                result.prediction, PredictionEnum.LIKELY_AUTHENTIC,
                f"Service failed for {filename}: got {result.prediction}"
            )
            self.assertGreaterEqual(result.confidence, 0.80)
            
            # 2. Test Production API POST /analyze/image
            resp = self.client.post("/analyze/image", files={"file": (filename, image_bytes, "image/jpeg")})
            self.assertEqual(resp.status_code, 200)
            api_result = resp.json()["result"]
            self.assertEqual(
                api_result["prediction"], "likely_authentic",
                f"API failed for {filename}: got {api_result['prediction']}"
            )

            # 3. Test Production Lane Detector (detectors/image.py)
            evidence = self.lane_detector.analyse(image_bytes, {"filename": filename})
            self.assertEqual(
                evidence.verdict, VerdictEnum.LIKELY_AUTHENTIC,
                f"Lane detector failed for {filename}: got {evidence.verdict}"
            )

    def test_compressed_images_trigger_uncertain(self):
        compressed_files = [
            "compressed_image.jpg",
            "normal_camera_photo.jpg"
        ]
        for filename in compressed_files:
            image_bytes = get_image_bytes(filename)
            
            # Service check
            result = self.service.analyze_image(image_bytes, filename=filename)
            self.assertEqual(
                result.prediction, PredictionEnum.UNCERTAIN,
                f"Service expected uncertain for {filename}, got {result.prediction}"
            )
            
            # API check
            resp = self.client.post("/analyze/image", files={"file": (filename, image_bytes, "image/jpeg")})
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["result"]["prediction"], "uncertain")

            # Lane detector check
            evidence = self.lane_detector.analyse(image_bytes, {"filename": filename})
            self.assertEqual(evidence.verdict, VerdictEnum.UNCERTAIN)

    def test_c2pa_ai_provenance_triggers_synthetic_verdict(self):
        filename = "c2pa_midjourney_art_1.jpg"
        image_bytes = get_image_bytes(filename)
        
        # Service check
        result = self.service.analyze_image(image_bytes, filename=filename)
        self.assertIn(result.prediction, [PredictionEnum.DECLARED_AI, PredictionEnum.LIKELY_SYNTHETIC])
        self.assertTrue(result.provenance.c2pa_found)

        # API check
        resp = self.client.post("/analyze/image", files={"file": (filename, image_bytes, "image/jpeg")})
        self.assertEqual(resp.status_code, 200)
        self.assertIn(resp.json()["result"]["prediction"], ["declared_ai", "likely_synthetic"])

        # Lane detector check
        evidence = self.lane_detector.analyse(image_bytes, {"filename": filename})
        self.assertEqual(evidence.verdict, VerdictEnum.DECLARED_AI)


if __name__ == "__main__":
    unittest.main()
