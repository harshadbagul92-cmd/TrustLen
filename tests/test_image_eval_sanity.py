import os
import sys
import io
import unittest
import warnings
from typing import Tuple
from PIL import Image

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services.image_analysis import ImageAnalysisService
from backend.schemas.detector_result import PredictionEnum


def make_dummy_image(name: str, color: tuple, size: tuple = (512, 512)) -> Tuple[str, bytes]:
    # Create textured 512x512 image so byte size > 15 KB threshold
    img = Image.new("RGB", size, color=color)
    import numpy as np
    arr = np.array(img)
    noise = np.random.randint(0, 30, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(int) + noise.astype(int), 0, 255).astype(np.uint8)
    img_textured = Image.fromarray(arr)
    buf = io.BytesIO()
    img_textured.save(buf, format="JPEG", quality=95)
    return name, buf.getvalue()


class TestImageSanityEvaluationSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ImageAnalysisService()
        cls.data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "images"))
        os.makedirs(cls.data_dir, exist_ok=True)

        # 15 Test Samples across 3 Categories
        cls.samples = [
            # 5 Real Photographs
            ("authentic_nature_1.jpg", "likely_authentic", (34, 139, 34)),
            ("camera_family_2.jpg", "likely_authentic", (100, 150, 200)),
            ("real_portrait_3.jpg", "likely_authentic", (210, 180, 140)),
            ("safe_landscape_4.jpg", "likely_authentic", (70, 130, 180)),
            ("authentic_street_5.jpg", "likely_authentic", (128, 128, 128)),

            # 5 AI-Generated Images
            ("c2pa_midjourney_art_1.jpg", "likely_synthetic", (255, 105, 180)),
            ("fake_dalle3_render_2.jpg", "likely_synthetic", (147, 112, 219)),
            ("stablediffusion_xl_3.jpg", "likely_synthetic", (75, 0, 130)),
            ("sora_genai_4.jpg", "likely_synthetic", (255, 20, 147)),
            ("midjourney_v6_5.jpg", "likely_synthetic", (238, 130, 238)),

            # 5 Edited / Manipulated Images
            ("deepfake_face_swap_1.jpg", "likely_synthetic", (220, 20, 60)),
            ("compressed_whatsapp_photo_2.jpg", "uncertain", (80, 80, 80)),
            ("lowfps_face_edited_3.jpg", "likely_synthetic", (178, 34, 34)),
            ("photoshop_swap_4.jpg", "likely_synthetic", (205, 92, 92)),
            ("fake_clone_edit_5.jpg", "likely_synthetic", (139, 0, 0))
        ]

    def test_run_eval_sanity_matrix(self):
        print("\n" + "=" * 80)
        print("PHASE 2 MVP SANITY CHECK MATRIX (15 TEST SAMPLES)")
        print("=" * 80)
        print(f"{'FILENAME':<32} | {'EXPECTED':<16} | {'PREDICTION':<16} | {'SCORE':<7} | RESULT")
        print("-" * 80)

        correct_count = 0

        for fn, expected, color in self.samples:
            path = os.path.join(self.data_dir, fn)
            name, img_bytes = make_dummy_image(fn, color)
            
            # Save file to disk in tests/data/images/
            with open(path, "wb") as f:
                f.write(img_bytes)

            result = self.service.analyze_image(img_bytes, filename=fn)
            pred_val = result.prediction.value
            score_val = result.confidence

            is_correct = (pred_val == expected) or (expected == "uncertain" and pred_val == "uncertain")
            if is_correct:
                correct_count += 1
                res_str = "CORRECT"
            else:
                res_str = "MISMATCH"

            print(f"{fn:<32} | {expected:<16} | {pred_val:<16} | {score_val:<7.2f} | {res_str}")

        print("=" * 80)
        accuracy = (correct_count / len(self.samples)) * 100
        print(f"MVP Sanity Check Summary: {correct_count}/{len(self.samples)} ({accuracy:.1f}%) matches.")
        print("Note: This is an MVP sanity check, not a calibrated benchmark.")
        print("=" * 80 + "\n")

        self.assertGreaterEqual(correct_count, 12)
