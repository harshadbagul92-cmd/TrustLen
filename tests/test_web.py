"""
Web layer tests: the HTTP API and the assets that make up the React interface.

These exist because the interface has a failure mode nothing else catches — a
missing vendor file, or a renamed endpoint — which leaves a blank screen while
every detector test still passes. The vendored libraries in particular must be
present and non-empty, since the whole point of vendoring them is that the app
keeps working when the venue wifi does not.
"""
import io
import json
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TRUSTLENS_NO_MODEL"] = "1"          # never call the network in tests

from fastapi.testclient import TestClient

import api
from trustlens import privacy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB = os.path.join(ROOT, "web")


def a_photo() -> bytes:
    rng = np.random.default_rng(5)
    arr = (rng.random((360, 360, 3)) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


class TestAssets(unittest.TestCase):
    """Every file the browser asks for must exist and be non-trivial."""

    # (path, minimum plausible size). An icon is legitimately tiny; a vendored
    # library that small means a failed download.
    REQUIRED = [
        ("index.html", 800), ("styles.css", 4000), ("app.js", 8000),
        ("favicon.svg", 200),
        ("vendor/react.production.min.js", 5000),
        ("vendor/react-dom.production.min.js", 50000),
        ("vendor/framer-motion.js", 50000),
        ("vendor/htm.js", 600),
    ]

    def test_all_assets_present_and_non_empty(self):
        for rel, minimum in self.REQUIRED:
            with self.subTest(asset=rel):
                path = os.path.join(WEB, rel)
                self.assertTrue(os.path.isfile(path), f"missing {rel}")
                self.assertGreater(os.path.getsize(path), minimum,
                                   f"{rel} looks truncated or failed to download")

    def test_index_references_only_local_scripts(self):
        """No CDN links: the interface must survive a dead network."""
        with open(os.path.join(WEB, "index.html"), encoding="utf-8") as fh:
            html = fh.read()
        for marker in ("unpkg.com", "cdn.jsdelivr", "cdnjs.cloudflare", "//cdn."):
            self.assertNotIn(marker, html, f"index.html still points at {marker}")
        self.assertIn("/static/vendor/react.production.min.js", html)

    def test_app_js_has_no_obvious_syntax_breakage(self):
        """Cheap guard against an unbalanced edit shipping a blank page."""
        with open(os.path.join(WEB, "app.js"), encoding="utf-8") as fh:
            src = fh.read()
        self.assertEqual(src.count("("), src.count(")"), "unbalanced parentheses in app.js")
        self.assertEqual(src.count("{"), src.count("}"), "unbalanced braces in app.js")
        self.assertIn("ReactDOM.createRoot", src)


class TestHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = TestClient(api.app)

    def test_index_is_served(self):
        r = self.c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("TrustLens", r.text)

    def test_static_assets_are_served(self):
        for rel in ("styles.css", "app.js", "vendor/htm.js"):
            with self.subTest(asset=rel):
                r = self.c.get(f"/static/{rel}")
                self.assertEqual(r.status_code, 200, rel)
                self.assertGreater(len(r.content), 400, rel)

    def test_health_reports_capabilities(self):
        body = self.c.get("/health").json()
        self.assertEqual(body["status"], "ok")
        for key in ("version", "language_model", "audio_video_decoder", "verdict_bands"):
            self.assertIn(key, body)

    def test_privacy_endpoint_states_no_retention(self):
        body = self.c.get("/privacy").json()
        self.assertFalse(body["stores_uploads"])
        self.assertFalse(body["stores_results"])
        self.assertEqual(body["scratch_files"]["files_remaining"], 0)

    def test_text_analysis_round_trip(self):
        r = self.c.post("/analyze/text", json={
            "text": "URGENT: account suspended, verify at http://bit.ly/x and send the OTP now.",
            "explain_with_model": False})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["verdict"], "likely_manipulated")
        self.assertTrue(body["explanation"])
        self.assertTrue(body["signals"])

    def test_image_upload_round_trip(self):
        r = self.c.post("/analyze",
                        files={"file": ("x.jpg", a_photo(), "image/jpeg")},
                        data={"explain_with_model": "false"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["modality"], "image")
        # The interface draws boxes from these keys; they must always exist.
        for key in ("noise_regions", "face_boxes"):
            self.assertIn(key, body["findings"])

    def test_every_response_carries_what_the_ui_renders(self):
        """The interface reads these fields unconditionally. A missing one is a
        blank panel, not an exception, so assert them explicitly."""
        r = self.c.post("/analyze/text", json={"text": "Just saying hello, nothing special here.",
                                               "explain_with_model": False})
        body = r.json()
        for key in ("modality", "verdict", "score", "confidence", "signals",
                    "reliability", "findings", "explanation", "elapsed_ms"):
            self.assertIn(key, body, f"response is missing {key}")
        self.assertEqual(len(body["confidence"]), 2)
        for s in body["signals"]:
            self.assertIn(s["direction"],
                          {"supports_manipulation", "supports_authentic", "context", "limitation"})

    def test_empty_request_is_rejected_cleanly(self):
        r = self.c.post("/analyze", data={"explain_with_model": "false"})
        self.assertEqual(r.status_code, 400)

    def test_garbage_upload_abstains_rather_than_500(self):
        r = self.c.post("/analyze",
                        files={"file": ("broken.jpg", b"not an image at all", "image/jpeg")},
                        data={"explain_with_model": "false"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["verdict"], "uncertain")

    def test_nothing_is_left_on_disk_afterwards(self):
        privacy.purge()
        self.c.post("/analyze", files={"file": ("x.jpg", a_photo(), "image/jpeg")},
                    data={"explain_with_model": "false"})
        self.assertEqual(privacy.residue(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
