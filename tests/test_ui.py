"""
Interface tests, using Streamlit's own headless runner.

These catch the class of bug that unit tests miss entirely: the app importing
a name that no longer exists, a verdict colour lookup with a missing key, or
the results panel crashing on a lane that returned an unusual shape. All of
those ship a blank red error page to a user while every unit test still passes.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Tests must not call the network: they would be slow, non-deterministic, and
# would burn the free Gemini quota that the live app needs. Every model path is
# exercised separately by hand.
os.environ["TRUSTLENS_NO_MODEL"] = "1"

try:
    from streamlit.testing.v1 import AppTest
    HAVE_APPTEST = True
except Exception:                                    # older streamlit
    HAVE_APPTEST = False

APP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))

SCAM = ("URGENT: Your bank account will be suspended within 24 hours due to unusual "
        "activity. Verify now at http://bit.ly/verify-acct-now and reply with the OTP "
        "we just sent.")
SAFE = "Hi, running about ten minutes late. Started the coffee already, see you shortly."


@unittest.skipUnless(HAVE_APPTEST, "streamlit.testing not available")
class TestInterface(unittest.TestCase):

    def _run(self, text=None):
        at = AppTest.from_file(APP, default_timeout=90).run()
        self.assertFalse(at.exception, f"app raised on first render: {at.exception}")
        if text is not None:
            at.text_area[0].set_value(text)
            at.button[0].click().run()
            self.assertFalse(at.exception, f"app raised after submit: {at.exception}")
        return at

    @staticmethod
    def _text(at) -> str:
        return "\n".join(m.value for m in at.markdown)

    def test_first_render_is_clean(self):
        at = self._run()
        body = self._text(at)
        self.assertIn("TrustLens", body)
        # The standing promise must be on screen before anything is submitted.
        self.assertIn("never answers with a flat", body)

    def test_scam_renders_verdict_and_quoted_evidence(self):
        at = self._run(SCAM)
        body = self._text(at)
        self.assertIn("Likely manipulated", body)
        self.assertIn("Points to manipulation", body)
        self.assertIn("bit.ly", body, "the matched text should be quoted back to the user")

    def test_safe_message_renders_the_authentic_band(self):
        at = self._run(SAFE)
        body = self._text(at)
        self.assertIn("No evidence of manipulation", body)
        self.assertNotIn("Likely manipulated", body)

    def test_limitations_are_always_offered(self):
        """A result must never imply a check ran that did not."""
        at = self._run(SCAM)
        labels = [e.label for e in at.expander]
        self.assertTrue(any("could not tell you" in l for l in labels),
                        f"no limitations panel in {labels}")

    def test_raw_evidence_is_always_available(self):
        at = self._run(SCAM)
        labels = [e.label for e in at.expander]
        self.assertTrue(any("Everything we measured" in l for l in labels))

    def test_privacy_promise_is_shown_with_the_result(self):
        at = self._run(SAFE)
        self.assertIn("stores nothing", self._text(at))

    def test_empty_submission_is_handled(self):
        at = AppTest.from_file(APP, default_timeout=90).run()
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(at.warning, "empty input should warn, not crash")


if __name__ == "__main__":
    unittest.main(verbosity=2)
