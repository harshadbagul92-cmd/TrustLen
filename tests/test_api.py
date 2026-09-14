import os
import sys
import unittest
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main import app


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "online")

    def test_analyze_text_form(self):
        payload = {
            "text": "URGENT: Account suspended! Pay immediately using $500 Apple gift card code."
        }
        response = self.client.post("/analyze", data=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "text")
        self.assertEqual(data["verdict"], "likely_manipulated")
        self.assertTrue(len(data["explanation"]) > 0)

    def test_analyze_json_endpoint(self):
        payload = {
            "text": "Dear PCCOE Students, Please submit your Computer Networks lab assignment 4 by Friday."
        }
        response = self.client.post("/analyze/json", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "text")
        self.assertEqual(data["verdict"], "likely_authentic")

    def test_analyze_image_upload(self):
        files = {"file": ("c2pa_midjourney_photo.jpg", b"fake_image_bytes", "image/jpeg")}
        response = self.client.post("/analyze", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "image")
        self.assertEqual(data["verdict"], "declared_ai")

    def test_analyze_audio_upload(self):
        files = {"file": ("elevenlabs_bank_scam.wav", b"fake_audio_pcm_bytes", "audio/wav")}
        response = self.client.post("/analyze", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "audio")
        self.assertEqual(data["verdict"], "declared_ai")

    def test_analyze_video_upload(self):
        files = {"file": ("video_face_swap.mp4", b"fake_video_mp4_bytes", "video/mp4")}
        response = self.client.post("/analyze", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modality"], "video")
        self.assertEqual(data["verdict"], "likely_manipulated")

    def test_analyze_missing_input_400(self):
        response = self.client.post("/analyze", data={})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
