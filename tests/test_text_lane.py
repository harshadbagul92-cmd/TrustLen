import os
import sys
import unittest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import VerdictEnum, ModalityEnum
from detectors.text import TextDetector


class TestTextLaneMemberA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = TextDetector()

    # ==================== 4 BLATANT SCAMS ====================

    def test_blatant_scam_1_irs_giftcard(self):
        text = (
            "URGENT FINAL NOTICE: Internal Revenue Service warrant issued for your arrest due to unpaid taxes. "
            "To avoid immediate police custody, pay immediately using $500 Apple gift card code."
        )
        evidence = self.detector.analyse(text)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_MANIPULATED)
        self.assertGreater(evidence.score, 0.65)
        signal_names = [s.name for s in evidence.signals]
        self.assertIn("rule_urgency_language", signal_names)
        self.assertIn("rule_payment_giftcard_demand", signal_names)

    def test_blatant_scam_2_lottery_wire_transfer(self):
        text = (
            "Congratulations! You have won $1,000,000 in the International Mega Draw Prize. "
            "Claim your cash prize now by sending a $250 wire transfer processing fee to our claims agent."
        )
        evidence = self.detector.analyse(text)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_MANIPULATED)
        self.assertGreater(evidence.score, 0.65)
        signal_names = [s.name for s in evidence.signals]
        self.assertIn("rule_prize_lottery_framing", signal_names)
        self.assertIn("rule_payment_giftcard_demand", signal_names)

    def test_blatant_scam_3_bank_otp_threat(self):
        text = (
            "URGENT: Your account has been suspended due to unauthorized access detected. "
            "Send your OTP verification code immediately to restore access within 24 hours."
        )
        evidence = self.detector.analyse(text)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_MANIPULATED)
        self.assertGreater(evidence.score, 0.65)
        signal_names = [s.name for s in evidence.signals]
        self.assertIn("rule_otp_password_request", signal_names)
        self.assertIn("rule_account_closure_threat", signal_names)

    def test_blatant_scam_4_whatsapp_emergency(self):
        text = (
            "Hi Mom, I lost my phone and wallet in an emergency. Respond within 2 hours! "
            "Please send $400 via Zelle immediately so I can pay the clinic fee."
        )
        evidence = self.detector.analyse(text)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_MANIPULATED)
        self.assertGreater(evidence.score, 0.65)
        signal_names = [s.name for s in evidence.signals]
        self.assertIn("rule_payment_giftcard_demand", signal_names)

    # ==================== 4 SUBTLE SCAMS / PHISHING ====================

    def test_subtle_scam_1_docusign_digitswap(self):
        text = "Please review and sign your updated employment agreement at http://docus1gn-verify-auth.com."
        evidence = self.detector.analyse(text)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_MANIPULATED)
        signal_names = [s.name for s in evidence.signals]
        self.assertIn("rule_punycode_digitswap_domain", signal_names)

    def test_subtle_scam_2_shortened_link_utility(self):
        text = "Your utility invoice is ready. Avoid late service restriction by viewing your bill now: http://bit.ly/3xX9aQz."
        evidence = self.detector.analyse(text)
        self.assertIn(evidence.verdict, [VerdictEnum.LIKELY_MANIPULATED, VerdictEnum.UNCERTAIN])
        signal_names = [s.name for s in evidence.signals]
        self.assertIn("rule_shortened_url", signal_names)

    def test_subtle_scam_3_display_name_mismatch(self):
        text = "Security Alert: Someone logged into your account."
        metadata = {"headers": {"From": '"PayPal Support" <notice@security-update-center.xyz>'}}
        evidence = self.detector.analyse(text, metadata=metadata)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_MANIPULATED)
        signal_names = [s.name for s in evidence.signals]
        self.assertIn("rule_display_name_mismatch", signal_names)

    def test_subtle_scam_4_punycode_domain(self):
        text = "Security check required for your Microsoft account: visit http://xn--microsft-v3a.com to confirm."
        evidence = self.detector.analyse(text)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_MANIPULATED)
        signal_names = [s.name for s in evidence.signals]
        self.assertIn("rule_punycode_digitswap_domain", signal_names)

    # ==================== 4 SAFE MESSAGES ====================

    def test_safe_1_pccoe_assignment(self):
        text = (
            "Dear PCCOE Students, Please remember to submit your Computer Networks lab assignment 4 "
            "on the college portal by Friday 5 PM. Contact your TA if you have any questions."
        )
        evidence = self.detector.analyse(text)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_AUTHENTIC)
        self.assertLess(evidence.score, 0.35)

    def test_safe_2_github_password_reset(self):
        text = (
            "We received a request to reset your GitHub password. If you initiated this request, "
            "you can reset your password directly on github.com inside your account settings."
        )
        evidence = self.detector.analyse(text)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_AUTHENTIC)
        self.assertLess(evidence.score, 0.35)

    def test_safe_3_amazon_tracking(self):
        text = (
            "Your Amazon order #408-19283-11 has been shipped! Your package is scheduled to arrive tomorrow. "
            "You can track your delivery status directly inside the Amazon mobile app."
        )
        evidence = self.detector.analyse(text)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_AUTHENTIC)
        self.assertLess(evidence.score, 0.35)

    def test_safe_4_friendly_coffee(self):
        text = "Hey Harshad, are you free this Saturday afternoon? Let's grab a coffee at the campus canteen and discuss our project."
        evidence = self.detector.analyse(text)
        self.assertEqual(evidence.verdict, VerdictEnum.LIKELY_AUTHENTIC)
        self.assertLess(evidence.score, 0.35)


if __name__ == "__main__":
    unittest.main()
