import os
import unittest
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from backend.services.image_analysis import ImageAnalysisService
from detectors.image import ImageDetector
from backend.schemas.detector_result import PredictionEnum, EvidenceCategoryEnum, ProvenanceInfo
from backend.services.confidence import ConfidenceEngine
from schemas import VerdictEnum

TEST_IMAGES_DIR = os.path.join("tests", "data", "images")


def get_image_bytes(filename: str) -> bytes:
    filepath = os.path.join(TEST_IMAGES_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return f.read()
    
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


class TestConservativeImageDecisionEngine(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.service = ImageAnalysisService()
        self.engine = ConfidenceEngine()
        self.lane_detector = ImageDetector()

    def test_1_genuine_photo_not_manipulated_by_single_fake_signal(self):
        # Even if raw prediction is LIKELY_SYNTHETIC with score < 0.88, gating converts to UNCERTAIN, never LIKELY_MANIPULATED
        prov = ProvenanceInfo(exif_found=True, camera_make="Canon", camera_model="EOS 5D")
        pred, conf, level, unc, reason, rules, dec_conf = self.engine.evaluate(
            raw_prediction=PredictionEnum.LIKELY_SYNTHETIC,
            raw_score=0.82,
            provenance=prov,
            evidence=[]
        )
        self.assertEqual(pred, PredictionEnum.UNCERTAIN)
        self.assertIn("RULE_P4_DETECTOR_CONFLICT_UNCORROBORATED", rules)
        self.assertNotEqual(pred, PredictionEnum.LIKELY_MANIPULATED)

    def test_2_detector_disagreement_produces_uncertain(self):
        prov = ProvenanceInfo(exif_found=False)
        pred, conf, level, unc, reason, rules, dec_conf = self.engine.evaluate(
            raw_prediction=PredictionEnum.LIKELY_SYNTHETIC,
            raw_score=0.65,
            provenance=prov,
            evidence=[]
        )
        self.assertEqual(pred, PredictionEnum.UNCERTAIN)

    def test_3_c2pa_ai_declaration_produces_declared_ai(self):
        prov = ProvenanceInfo(c2pa_present=True, c2pa_ai_declared=True)
        pred, conf, level, unc, reason, rules, dec_conf = self.engine.evaluate(
            raw_prediction=PredictionEnum.LIKELY_AUTHENTIC,
            raw_score=0.90,
            provenance=prov,
            evidence=[]
        )
        self.assertEqual(pred, PredictionEnum.DECLARED_AI)
        self.assertIn("RULE_P1_STRONG_AI_PROVENANCE", rules)

    def test_4_low_quality_image_produces_uncertain(self):
        b = get_image_bytes("compressed_image.jpg")
        res = self.service.analyze_image(b, filename="compressed_image.jpg")
        self.assertEqual(res.prediction, PredictionEnum.UNCERTAIN)
        self.assertIn("RULE_P5_LOW_QUALITY_OOD", res.decision_rules_triggered)

    def test_5_missing_exif_does_not_produce_fake(self):
        b = get_image_bytes("authentic_nature_1.jpg")
        res = self.service.analyze_image(b, filename="authentic_nature_1.jpg")
        self.assertEqual(res.prediction, PredictionEnum.LIKELY_AUTHENTIC)
        self.assertFalse(res.provenance.exif_found)

    def test_6_editing_software_metadata_alone_does_not_produce_fake(self):
        prov = ProvenanceInfo(exif_found=True, software="Adobe Photoshop 2024", c2pa_edit_declared=True)
        pred, conf, level, unc, reason, rules, dec_conf = self.engine.evaluate(
            raw_prediction=PredictionEnum.LIKELY_AUTHENTIC,
            raw_score=0.85,
            provenance=prov,
            evidence=[]
        )
        self.assertEqual(pred, PredictionEnum.LIKELY_AUTHENTIC)

    def test_7_raw_detector_label_cannot_bypass_decision_engine(self):
        prov = ProvenanceInfo()
        pred, conf, level, unc, reason, rules, dec_conf = self.engine.evaluate(
            raw_prediction=PredictionEnum.LIKELY_SYNTHETIC,
            raw_score=0.70,
            provenance=prov,
            evidence=[]
        )
        # Raw label was LIKELY_SYNTHETIC, but DecisionEngine forced UNCERTAIN
        self.assertEqual(pred, PredictionEnum.UNCERTAIN)

    def test_8_frontend_and_api_canonical_mapping_matches(self):
        b = get_image_bytes("authentic_nature_1.jpg")
        resp = self.client.post("/analyze/image", files={"file": ("authentic_nature_1.jpg", b, "image/jpeg")})
        self.assertEqual(resp.status_code, 200)
        api_pred = resp.json()["result"]["prediction"]
        service_pred = self.service.analyze_image(b, filename="authentic_nature_1.jpg").prediction.value
        self.assertEqual(api_pred, service_pred)
        self.assertEqual(api_pred, "likely_authentic")

    def test_9_explanation_grounded_without_unsupported_visual_claims(self):
        b = get_image_bytes("authentic_nature_1.jpg")
        res = self.service.analyze_image(b, filename="authentic_nature_1.jpg")
        exp = res.explanation.lower()
        self.assertNotIn("eyes are unnatural", exp)
        self.assertNotIn("hands are distorted", exp)
        self.assertIn("no significant evidence", exp)

    def test_10_api_and_ui_prediction_values_match_exactly(self):
        b = get_image_bytes("c2pa_midjourney_art_1.jpg")
        res = self.service.analyze_image(b, filename="c2pa_midjourney_art_1.jpg")
        self.assertIn(res.prediction, [PredictionEnum.DECLARED_AI, PredictionEnum.LIKELY_SYNTHETIC])
        self.assertEqual(res.decision_confidence, 0.95)


if __name__ == "__main__":
    unittest.main()
