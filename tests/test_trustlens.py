"""
TrustLens test suite.

    python -m unittest discover tests -v

Every fixture is generated here at run time. Nothing depends on files in the
repository, so this passes on a fresh clone with no sample data committed.

The tests that matter most are the ones in TestNoFabrication. An earlier
version of this project scored images on their filename and invented Grad-CAM
regions for any picture containing a face. Those are regressions now.
"""
import io
import os
import subprocess
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Tests must not call the network: they would be slow, non-deterministic, and
# would burn the free Gemini quota that the live app needs. Every model path is
# exercised separately by hand.
os.environ["TRUSTLENS_NO_MODEL"] = "1"

from trustlens import detectors, privacy
from trustlens.detectors import forensics_audio as fa
from trustlens.detectors import forensics_image as fx
from trustlens.detectors import media
from trustlens.explainer import explain
from trustlens.schemas import Direction, Modality, Verdict


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def textured_photo(size=(480, 480), seed=3) -> bytes:
    """A JPEG with enough detail to pass the flatness gate."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, size[0], dtype=np.float32)
    img = np.stack([np.tile(x, (size[1], 1)),
                    np.tile(x[:, None], (1, size[0])),
                    np.full((size[1], size[0]), 128.0)], axis=2)
    img += rng.normal(0, 16, img.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def spliced_photo() -> tuple[bytes, tuple]:
    """A photo with a patch from a DIFFERENT image pasted in. Returns
    (bytes, ground-truth box)."""
    host = Image.open(io.BytesIO(textured_photo(seed=3))).convert("RGB")
    donor = Image.open(io.BytesIO(textured_photo(seed=99))).convert("RGB")
    # Re-encode the donor patch hard so it carries its own compression history.
    patch = donor.crop((0, 0, 160, 140))
    tmp = io.BytesIO()
    patch.save(tmp, "JPEG", quality=25)
    tmp.seek(0)
    patch = Image.open(tmp).convert("RGB")
    box = (60, 70, 60 + 160, 70 + 140)
    host.paste(patch, (60, 70))
    out = io.BytesIO()
    host.save(out, "JPEG", quality=90)
    return out.getvalue(), box


def tone_wav(seconds=3.0, sr=16000, freq=220.0, pad_silence=False) -> bytes:
    import wave
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sig = 0.4 * np.sin(2 * np.pi * freq * t)
    sig += np.random.default_rng(1).normal(0, 0.004, sig.shape)   # noise floor
    if pad_silence:
        sig = np.concatenate([sig, np.zeros(int(sr * seconds))])  # exact zeros
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def small_video() -> bytes | None:
    exe = media.ffmpeg_path()
    if not exe:
        return None
    proc = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4",
         "-movflags", "frag_keyframe+empty_moov", "pipe:1"],
        capture_output=True)
    return proc.stdout if proc.returncode == 0 and proc.stdout else None


SCAM = ("URGENT: Your account will be suspended within 24 hours. Verify at "
        "http://bit.ly/verify-now and reply with the OTP we sent.")
SAFE = ("Hi, running about ten minutes late for the meeting. Started the coffee "
        "already, see you shortly.")


# ---------------------------------------------------------------------------
class TestNoFabrication(unittest.TestCase):
    """The system must not manufacture evidence it did not measure."""

    def test_filename_cannot_change_the_verdict(self):
        """REGRESSION: the image lane used to score on the filename, so a file
        called deepfake.jpg was given 0.86 automatically."""
        data = textured_photo()
        plain = detectors.analyse(data, "holiday.jpg", "image/jpeg")
        loaded = detectors.analyse(data, "obvious_deepfake_midjourney_fake_swap.jpg", "image/jpeg")
        self.assertEqual(plain.score, loaded.score)
        self.assertEqual(plain.verdict, loaded.verdict)

    def test_audio_filename_cannot_change_the_verdict(self):
        """REGRESSION: the audio lane used to report vocoder artifacts for any
        file whose name contained 'clone', without decoding a single byte."""
        data = tone_wav()
        plain = detectors.analyse(data, "recording.wav", "audio/wav")
        loaded = detectors.analyse(data, "elevenlabs_voice_clone_deepfake.wav", "audio/wav")
        self.assertEqual(plain.score, loaded.score)
        self.assertEqual([s.name for s in plain.signals], [s.name for s in loaded.signals])

    def test_every_signal_carries_a_measured_value(self):
        for label, data, name in (("image", textured_photo(), "a.jpg"),
                                  ("audio", tone_wav(), "a.wav")):
            with self.subTest(lane=label):
                ev = detectors.analyse(data, name, "")
                self.assertTrue(ev.signals)
                for s in ev.signals:
                    self.assertIsNotNone(s.value, f"{s.name} has no measured value")
                    self.assertTrue(s.human.strip(), f"{s.name} has no plain-English text")

    def test_flat_image_abstains_instead_of_accusing(self):
        """REGRESSION: a near-black frame used to produce confident
        'manipulation' regions, because dividing by near-zero texture explodes
        the z-scores."""
        buf = io.BytesIO()
        Image.new("RGB", (400, 400), (6, 6, 8)).save(buf, "JPEG", quality=90)
        ev = detectors.analyse(buf.getvalue(), "dark.jpg", "image/jpeg")
        self.assertNotEqual(ev.verdict, Verdict.LIKELY_MANIPULATED)
        self.assertEqual([s for s in ev.signals if "localized" in s.name], [])

    def test_explainer_will_not_invent_a_technique(self):
        """The explainer must not call something a face swap when no signal did."""
        from trustlens.explainer import _grounded
        ev = detectors.analyse(textured_photo(), "a.jpg", "image/jpeg")
        self.assertFalse(_grounded("We detected a face swap using GAN artifacts. " * 3, ev))


class TestRouting(unittest.TestCase):
    def test_extension_routing(self):
        cases = {"a.jpg": Modality.IMAGE, "a.png": Modality.IMAGE,
                 "a.mp3": Modality.AUDIO, "a.wav": Modality.AUDIO,
                 "a.mp4": Modality.VIDEO, "a.mkv": Modality.VIDEO,
                 "notes": Modality.TEXT}
        for name, expected in cases.items():
            self.assertEqual(detectors.detect_modality(name), expected, name)

    def test_content_type_beats_extension(self):
        self.assertEqual(detectors.detect_modality("thing.bin", "audio/mpeg"), Modality.AUDIO)


class TestTextLane(unittest.TestCase):
    def test_scam_is_flagged_with_quoted_evidence(self):
        ev = detectors.get(Modality.TEXT).analyse(SCAM, {})
        self.assertEqual(ev.verdict, Verdict.LIKELY_MANIPULATED)
        hits = ev.by_direction(Direction.MANIPULATION)
        self.assertTrue(hits)
        self.assertTrue(any("bit.ly" in str(s.value) for s in hits),
                        "the matched text should be quoted back")

    def test_ordinary_message_is_not_flagged(self):
        ev = detectors.get(Modality.TEXT).analyse(SAFE, {})
        self.assertEqual(ev.verdict, Verdict.LIKELY_AUTHENTIC)

    def test_short_input_abstains(self):
        ev = detectors.get(Modality.TEXT).analyse("hey", {})
        self.assertEqual(ev.verdict, Verdict.UNCERTAIN)
        self.assertIn("text_too_short", ev.reliability.ood_flags)


class TestImageLane(unittest.TestCase):
    def test_clean_photo_is_not_flagged(self):
        ev = detectors.analyse(textured_photo(), "a.jpg", "image/jpeg")
        self.assertNotEqual(ev.verdict, Verdict.LIKELY_MANIPULATED)

    def test_splice_is_localized_near_the_truth(self):
        data, truth = spliced_photo()
        ev = detectors.analyse(data, "a.jpg", "image/jpeg")
        regions = (ev.findings.get("noise_regions") or []) + (ev.findings.get("ela_regions") or [])
        self.assertTrue(regions, "no anomalous region found in a spliced image")

        def overlaps(a, b):
            return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
        self.assertTrue(any(overlaps(tuple(r["bbox"]), truth) for r in regions),
                        f"found {[r['bbox'] for r in regions]}, none overlapping {truth}")

    def test_jpeg_quality_inversion(self):
        img = Image.new("RGB", (256, 256))
        d = ImageDraw.Draw(img)
        for i in range(0, 256, 8):
            d.line([(i, 0), (i, 256)], fill=(i, 255 - i, (i * 3) % 256), width=3)
        for target in (60, 75, 90):
            with self.subTest(quality=target):
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=target)
                buf.seek(0)
                got = fx.jpeg_quality_from_qtables(Image.open(buf))
                self.assertIsNotNone(got)
                self.assertLessEqual(abs(got - target), 4)

    def test_undecodable_file_abstains(self):
        ev = detectors.analyse(b"this is not an image", "broken.jpg", "image/jpeg")
        self.assertEqual(ev.verdict, Verdict.UNCERTAIN)
        self.assertTrue(ev.reliability.ood_flags)


@unittest.skipUnless(media.ffmpeg_available(), "ffmpeg not available")
class TestAudioLane(unittest.TestCase):
    def test_tone_decodes_and_measures(self):
        ev = detectors.analyse(tone_wav(), "a.wav", "audio/wav")
        self.assertEqual(ev.modality, Modality.AUDIO)
        self.assertGreater(ev.findings["basic"]["duration_s"], 2.0)
        self.assertGreater(ev.findings["pitch"]["voiced_frames"], 0)

    def test_digital_silence_is_detected(self):
        ev = detectors.analyse(tone_wav(pad_silence=True), "a.wav", "audio/wav")
        self.assertGreater(ev.findings["digital_silence_pct"], 10.0)
        self.assertTrue(any(s.name == "digital_exact_silence" for s in ev.signals))

    def test_silence_threshold_is_relative_to_loud_parts(self):
        """REGRESSION: a steady-amplitude recording was reported as 100% silence."""
        sr = 16000
        steady = 0.3 * np.sin(2 * np.pi * 200 * np.arange(sr * 2) / sr).astype(np.float32)
        self.assertLess(fa.silence_structure(steady, sr)["silence_pct"], 5.0)

    def test_too_short_abstains(self):
        ev = detectors.analyse(tone_wav(seconds=0.2), "a.wav", "audio/wav")
        self.assertEqual(ev.verdict, Verdict.UNCERTAIN)

    def test_always_states_the_voice_clone_limitation(self):
        ev = detectors.analyse(tone_wav(), "a.wav", "audio/wav")
        self.assertTrue(any(s.name == "no_voice_clone_model" for s in ev.limitations))


@unittest.skipUnless(media.ffmpeg_available(), "ffmpeg not available")
class TestVideoLane(unittest.TestCase):
    def test_frames_are_read_and_reported(self):
        data = small_video()
        if not data:
            self.skipTest("could not build a test video")
        ev = detectors.analyse(data, "a.mp4", "video/mp4")
        self.assertEqual(ev.modality, Modality.VIDEO)
        self.assertGreater(ev.findings.get("frames_sampled", 0), 0)

    def test_video_leaves_no_file_behind(self):
        """The privacy guarantee: OpenCV needs a path, so a scratch file is
        written - and must be gone by the time the result comes back."""
        data = small_video()
        if not data:
            self.skipTest("could not build a test video")
        privacy.purge()
        detectors.analyse(data, "a.mp4", "video/mp4")
        self.assertEqual(privacy.residue(), [], "a scratch file survived the analysis")


class TestPrivacy(unittest.TestCase):
    def test_scratch_file_is_removed_even_on_exception(self):
        seen = {}
        with self.assertRaises(RuntimeError):
            with privacy.scratch_file(b"sensitive bytes", ".bin") as path:
                seen["path"] = path
                self.assertTrue(os.path.exists(path))
                raise RuntimeError("boom")
        self.assertFalse(os.path.exists(seen["path"]))

    def test_no_residue_after_purge(self):
        privacy.purge()
        self.assertEqual(privacy.residue(), [])


class TestExplainer(unittest.TestCase):
    def test_template_states_scope_for_authentic_results(self):
        ev = detectors.analyse(textured_photo(), "a.jpg", "image/jpeg")
        text = explain(ev, use_model=False)
        self.assertIn("no evidence of manipulation", text.lower())
        self.assertNotIn("this is real", text.lower())

    def test_every_lane_produces_a_usable_explanation(self):
        for data, name in ((textured_photo(), "a.jpg"), (tone_wav(), "a.wav")):
            with self.subTest(name=name):
                ev = detectors.analyse(data, name, "")
                self.assertGreater(len(explain(ev, use_model=False)), 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)


AI_ISH = ("It is important to note that artificial intelligence plays a crucial role in "
          "the modern landscape. Furthermore, these systems underscore the multifaceted "
          "nature of digital innovation. Moreover, organisations must navigate the "
          "ever-evolving environment with care. In conclusion, this remains a testament "
          "to human progress and a beacon of what is possible.")
HUMAN_ISH = ("ok so i finally tried that cafe near the station. honestly? bit overpriced "
             "lol. the croissant was good though, flaky and warm. took ages to get a "
             "table tho - like 25 min?? anyway we're going back saturday if you're free")


class TestAiWrittenText(unittest.TestCase):
    """
    The text lane answers two separate questions: is this a scam, and was it
    written by a machine. A person can write a scam and a model can write
    something harmless, so neither answer may stand in for the other.
    """

    def _run(self, text):
        return detectors.get(Modality.TEXT).analyse(text, {})

    def test_human_written_scam_is_headlined_as_a_scam(self):
        """Not as 'no AI found' - that would bury the thing that matters."""
        ev = self._run(SCAM)
        self.assertEqual(ev.findings["headline"], "scam")
        self.assertGreater(ev.findings["scam_signal_count"], 0)

    def test_assistant_style_prose_is_headlined_as_ai(self):
        ev = self._run(AI_ISH)
        self.assertEqual(ev.findings["headline"], "ai_generated")
        self.assertGreater(ev.findings["ai_signal_count"], 0)

    def test_ordinary_human_writing_is_headlined_clean(self):
        ev = self._run(HUMAN_ISH)
        self.assertEqual(ev.findings["headline"], "clean")
        self.assertEqual(ev.findings["scam_signal_count"], 0)

    def test_every_text_result_states_the_reliability_limit(self):
        """AI-text detection is not a solved problem, and the result must say so
        every time - it is the difference between a hint and an accusation."""
        for text in (SCAM, SAFE, AI_ISH, HUMAN_ISH):
            with self.subTest(text=text[:30]):
                ev = self._run(text)
                self.assertTrue(any(s.name == "ai_text_detection_limits" for s in ev.limitations))
                self.assertIn("ai_text_detection_is_unreliable", ev.reliability.soft_flags)

    def test_style_score_alone_can_never_convict(self):
        """Stylometry is weak evidence and is capped so it cannot, by itself,
        push a passage past the 'likely manipulated' threshold."""
        from trustlens.detectors import stylometry
        m = stylometry.measure(AI_ISH)
        self.assertLessEqual(stylometry.reads_as_machine(m)["score"], 0.45)

    def test_burstiness_separates_even_prose_from_varied_prose(self):
        from trustlens.detectors import stylometry
        even = stylometry.measure(AI_ISH).get("burstiness")
        varied = stylometry.measure(HUMAN_ISH).get("burstiness")
        self.assertIsNotNone(even)
        self.assertIsNotNone(varied)
        self.assertLess(even, varied, "assistant-style prose should vary less than casual writing")

    def test_short_text_declines_to_judge_style(self):
        ev = self._run("Call me back when you can.")
        self.assertTrue(any(s.name == "too_short_for_style" for s in ev.limitations))
