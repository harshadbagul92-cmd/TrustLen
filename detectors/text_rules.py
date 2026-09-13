import re
from typing import List, Tuple, Dict, Any, Optional
from schemas import Signal


class TextRuleExtractor:
    """
    Member A - Rule Extractor Engine.
    Scans text messages and headers for 9 specific scam, phishing, and misinfo pattern classes.
    Returns structured Signal objects with exact matched spans and line numbers.
    """

    # 1. Urgency Language
    URGENCY_PATTERNS = [
        r"\b(?:urgent|act now|immediate action|respond within \d+ hours?|account will be closed in \d+ hours?|final notice|immediate response)\b",
        r"\b(?:suspended immediately|unauthorized access detected|security alert|take action now)\b"
    ]

    # 2. OTP / PIN / Password Requests
    OTP_PATTERNS = [
        r"\b(?:otp|one[- ]time password|verification code|pin|passcode|secret key|login credentials)\b",
        r"\b(?:send me the code|share your otp|enter your pin|verify password|confirm your login)\b"
    ]

    # 3. Payment or Gift Card Demands
    PAYMENT_PATTERNS = [
        r"\b(?:gift card|itunes card|amazon gift card|apple gift card|steam card|google play card)\b",
        r"\b(?:wire transfer|bitcoin|crypto|usdt|zelle|venmo|western union|cash app)\b",
        r"\b(?:pay immediately|transfer funds|send payment|unpaid fine|outstanding fee)\b"
    ]

    # 4. Shortened URLs & Known Shorteners
    SHORTENER_DOMAINS = [
        r"\b(?:bit\.ly|tinyurl\.com|t\.co|is\.gd|goo\.gl|buff\.ly|ow\.ly|rb\.gy|cutt\.ly)\b"
    ]

    # 5. Punycode and Digit-Swap Domains
    PUNYCODE_DIGITSWAP_PATTERNS = [
        r"\bxn--[a-z0-9-]+\.[a-z]{2,}\b",  # Punycode header
        r"\b[a-z0-9-]*(?:paypa1|banc0|g00gle|micr0soft|arnazon|docus1gn|secunty|wellsfarg0|chase-verify)[a-z0-9-]*\.[a-z]{2,}\b",
        r"\b[a-z0-9-]+\.(?:xyz|top|work|click|club|info|kim|country|stream|download)\b" # Suspicious TLDs
    ]

    # 6. Prize or Lottery Framing
    PRIZE_PATTERNS = [
        r"\b(?:congratulations|you have won|winner|lottery|draw prize|claim your \$?\d+|lucky winner|cash prize)\b",
        r"\b(?:claim reward|selected for a prize|free grant|inheritance payout)\b"
    ]

    # 7. Account Closure Threats
    ACCOUNT_CLOSURE_PATTERNS = [
        r"\b(?:account (?:has been )?(?:suspended|closure|closed|freeze|frozen|termination|terminated)|will be closed|deactivated|permanently locked)\b",
        r"\b(?:verify immediately or access will be restricted|account frozen)\b"
    ]

    def extract_signals(self, text: str, headers: Optional[Dict[str, str]] = None) -> List[Signal]:
        signals = []
        lines = text.splitlines()
        headers = headers or {}

        # 1. Urgency Language
        for idx, line in enumerate(lines, 1):
            for pat in self.URGENCY_PATTERNS:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    signals.append(Signal(
                        name="rule_urgency_language",
                        human=f"Urgency/High-pressure language detected: '{match.group(0)}'.",
                        value=match.group(0),
                        weight=0.80,
                        where=f"Line {idx}"
                    ))
                    break

        # 2. OTP/PIN/Password Requests
        for idx, line in enumerate(lines, 1):
            for pat in self.OTP_PATTERNS:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    signals.append(Signal(
                        name="rule_otp_password_request",
                        human=f"Request for sensitive authentication credential (OTP/PIN): '{match.group(0)}'.",
                        value=match.group(0),
                        weight=0.92,
                        where=f"Line {idx}"
                    ))
                    break

        # 3. Payment or Gift Card Demands
        for idx, line in enumerate(lines, 1):
            for pat in self.PAYMENT_PATTERNS:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    signals.append(Signal(
                        name="rule_payment_giftcard_demand",
                        human=f"Untraceable payment or gift-card demand detected: '{match.group(0)}'.",
                        value=match.group(0),
                        weight=0.90,
                        where=f"Line {idx}"
                    ))
                    break

        # 4. Shortened & Lookalike URLs
        for idx, line in enumerate(lines, 1):
            for pat in self.SHORTENER_DOMAINS:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    signals.append(Signal(
                        name="rule_shortened_url",
                        human=f"Shortened link masking final destination: '{match.group(0)}'.",
                        value=match.group(0),
                        weight=0.75,
                        where=f"Line {idx}"
                    ))
                    break

        # 5. Punycode & Digit-swap domains
        for idx, line in enumerate(lines, 1):
            for pat in self.PUNYCODE_DIGITSWAP_PATTERNS:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    signals.append(Signal(
                        name="rule_punycode_digitswap_domain",
                        human=f"Suspicious digit-swapped or lookalike domain: '{match.group(0)}'.",
                        value=match.group(0),
                        weight=0.95,
                        where=f"Line {idx}"
                    ))
                    break

        # 6. Prize or Lottery Framing
        for idx, line in enumerate(lines, 1):
            for pat in self.PRIZE_PATTERNS:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    signals.append(Signal(
                        name="rule_prize_lottery_framing",
                        human=f"Unsolicited prize or lottery claim framing: '{match.group(0)}'.",
                        value=match.group(0),
                        weight=0.85,
                        where=f"Line {idx}"
                    ))
                    break

        # 7. Account Closure Threats
        for idx, line in enumerate(lines, 1):
            for pat in self.ACCOUNT_CLOSURE_PATTERNS:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    signals.append(Signal(
                        name="rule_account_closure_threat",
                        human=f"Threat of account suspension or deactivation: '{match.group(0)}'.",
                        value=match.group(0),
                        weight=0.88,
                        where=f"Line {idx}"
                    ))
                    break

        # 8. Display-name vs Address Mismatch
        from_hdr = headers.get("From", "") or headers.get("from", "")
        if from_hdr:
            disp_match = re.search(r'["\']?([^"\'<]+)["\']?\s*<([^>]+)>', from_hdr)
            if disp_match:
                disp_name, email_addr = disp_match.group(1).strip(), disp_match.group(2).strip()
                domain = email_addr.split("@")[-1].lower() if "@" in email_addr else ""
                brands = ["paypal", "google", "microsoft", "amazon", "apple", "chase", "wells fargo", "pccoe", "bank"]
                matched_brand = next((b for b in brands if b in disp_name.lower()), None)
                if matched_brand and matched_brand not in domain:
                    signals.append(Signal(
                        name="rule_display_name_mismatch",
                        human=f"Display name '{disp_name}' does not match sender domain '@{domain}'.",
                        value=from_hdr,
                        weight=0.92,
                        where="Header: From"
                    ))

        # 9. Reply-To Mismatch
        reply_to = headers.get("Reply-To", "") or headers.get("reply-to", "")
        if from_hdr and reply_to:
            from_domain = from_hdr.split("@")[-1].replace(">", "").strip() if "@" in from_hdr else ""
            reply_domain = reply_to.split("@")[-1].replace(">", "").strip() if "@" in reply_to else ""
            if from_domain and reply_domain and from_domain != reply_domain:
                signals.append(Signal(
                    name="rule_reply_to_mismatch",
                    human=f"Reply-To domain '@{reply_domain}' differs from From domain '@{from_domain}'.",
                    value=f"From: {from_hdr} | Reply-To: {reply_to}",
                    weight=0.89,
                    where="Header: Reply-To"
                ))

        return signals
