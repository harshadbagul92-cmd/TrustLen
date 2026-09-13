import os
import sys
import unittest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import VerdictEnum, ModalityEnum
from detectors.audio import AudioDetector
from detectors.audio_aasist import AASISTProcessor


class TestAudioLaneMemberA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = AudioDetector()
        cls.aasist = AASISTProcessor()

    def test_1_aasist_windowing_max_mean_scores(self):
        fake_pcm = b"\x00\x00" * 16000 * 12  # 12 seconds of 16kHz audio
        max_s, mean_s, signals, stats = self.aasist.process(fake_pcm, metadata={"filename": "elevenlabs_test.wav"})
        
        self.assertGreater(stats["num_windows"], 1)
        self.assertGreaterEqual(max_s, mean_s)
        self.assertIn("worst_window_timestamp", stats)
        self.assertGreater(len(signals), 0)
        self.assertIn("Timestamp", signals[0].where)

    def test_2_supporting_acoustic_signals(self):
        fake_audio_bytes = b"sample_pcm_bytes"
        metadata = {"filename": "fake_clone_speech.wav"}
        evidence = self.detector.analyse(fake_audio_bytes, metadata=metadata)
        
        self.assertEqual(evidence.modality, ModalityEnum.AUDIO)
        signal_names = [s.name for s in evidence.signals]
        
        self.assertIn("unnatural_pitch_flatness_f0", signal_names)
        self.assertIn("unnatural_silence_respiration_ratio", signal_names)

    def test_3_dual_findings_cloned_voice_scam_script(self):
        fake_audio_bytes = b"elevenlabs_scam_audio_bytes"
        metadata = {"filename": "elevenlabs_bank_scam.wav"}
        evidence = self.detector.analyse(fake_audio_bytes, metadata=metadata)
        
        self.assertEqual(evidence.verdict, VerdictEnum.DECLARED_AI)
        self.assertGreater(evidence.score, 0.65)
        
        signal_names = [s.name for s in evidence.signals]
        # Finding 1: Acoustic synthesis finding
        self.assertTrue(any("aasist" in name or "pitch" in name or "spectral" in name for name in signal_names))
        # Finding 2: Transcribed scam script text finding from Phase 1 text lane
        self.assertTrue(any(name.startswith("transcript_") for name in signal_names))

    def test_4_authentic_audio_no_evidence_of_manipulation(self):
        fake_audio_bytes = b"authentic_speech_bytes"
        metadata = {"filename": "authentic_safe_meeting.wav"}
        evidence = self.detector.analyse(fake_audio_bytes, metadata=metadata)
        
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_AUTHENTIC)
        self.assertLess(evidence.score, 0.35)


if __name__ == "__main__":
    unittest.main()
