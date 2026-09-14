import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import VerdictEnum, ModalityEnum
from detectors.image import ImageDetector
from detectors.video import VideoDetector


class TestImageVideoLanesMemberB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_det = ImageDetector()
        cls.video_det = VideoDetector()

    # ==================== IMAGE LANE TESTS ====================

    def test_1_image_c2pa_declared_ai(self):
        meta = {"filename": "c2pa_midjourney_photo.jpg"}
        evidence = self.image_det.analyse(b"dummy_bytes", metadata=meta)
        self.assertEqual(evidence.modality, ModalityEnum.IMAGE)
        self.assertEqual(evidence.verdict, VerdictEnum.DECLARED_AI)
        self.assertGreater(evidence.score, 0.65)
        self.assertIn("c2pa_manifest", evidence.provenance)

    def test_2_image_deepfake_face_manipulated(self):
        meta = {"filename": "face_swap_deepfake.jpg"}
        evidence = self.image_det.analyse(b"dummy_bytes", metadata=meta)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_MANIPULATED)
        self.assertGreater(evidence.score, 0.65)
        signal_names = [s.name for s in evidence.signals]
        self.assertIn("facial_boundary_blending", signal_names)

    def test_3_image_jpeg_compression_ood_uncertain(self):
        meta = {"filename": "compressed_whatsapp_photo.jpg"}
        evidence = self.image_det.analyse(b"dummy_bytes", metadata=meta)
        self.assertEqual(evidence.verdict, VerdictEnum.UNCERTAIN)
        self.assertIn("severe_jpeg_compression", evidence.reliability.ood_flags)
        self.assertGreater(evidence.reliability.band, 0.0)

    def test_4_image_authentic_photo(self):
        meta = {"filename": "authentic_landscape_camera.jpg"}
        evidence = self.image_det.analyse(b"dummy_bytes", metadata=meta)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_AUTHENTIC)
        self.assertLess(evidence.score, 0.35)

    # ==================== VIDEO LANE TESTS ====================

    def test_5_video_sora_declared_ai(self):
        meta = {"filename": "sora_generative_clip.mp4"}
        evidence = self.video_det.analyse(b"dummy_bytes", metadata=meta)
        self.assertEqual(evidence.modality, ModalityEnum.VIDEO)
        self.assertEqual(evidence.verdict, VerdictEnum.DECLARED_AI)
        self.assertGreater(evidence.score, 0.65)

    def test_6_video_deepfake_swap(self):
        meta = {"filename": "video_face_swap.mp4"}
        evidence = self.video_det.analyse(b"dummy_bytes", metadata=meta)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_MANIPULATED)
        self.assertGreater(evidence.score, 0.65)
        signal_names = [s.name for s in evidence.signals]
        self.assertIn("lip_sync_audio_visual_mismatch", signal_names)

    def test_7_video_lowfps_ood_uncertain(self):
        meta = {"filename": "lowfps_dropped_clip.mp4"}
        evidence = self.video_det.analyse(b"dummy_bytes", metadata=meta)
        self.assertEqual(evidence.verdict, VerdictEnum.UNCERTAIN)
        self.assertIn("irregular_frame_rate", evidence.reliability.ood_flags)

    def test_8_video_authentic_clip(self):
        meta = {"filename": "authentic_meeting_recording.mp4"}
        evidence = self.video_det.analyse(b"dummy_bytes", metadata=meta)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_AUTHENTIC)
        self.assertLess(evidence.score, 0.35)


if __name__ == "__main__":
    unittest.main()
