"""
Text lane: scam, phishing and misinformation.

Three layers, each independently useful:

  1. Rules       deterministic red flags with the exact matched text quoted
                 back, so a person can see what triggered and judge for
                 themselves. Works with no API key and no network.
  2. Classifier  a language model reading the message AND the rule hits, so it
                 reasons over evidence rather than vibes.
  3. SelfCheck   for factual claims, sample the model N times and measure how
                 much the answers diverge (Manakul et al. 2023). Consistent
                 answers mean grounded knowledge; answers that contradict each
                 other mean the model is guessing, which is the signal that a
                 claim is not well-supported.

Deliberately NOT implemented: scoring on spelling or grammar. It is a real
correlate of some scams, but it penalises anyone writing in a second language,
and a fluent scam sails past it. The cost of that unfairness is not worth the
small accuracy gain.
"""
import re
from typing import Any, Dict, List, Tuple

from .. import config, llm, schemas
from ..schemas import Direction, Evidence, Modality, Reliability, Signal
from .base import Detector

# Each rule: (name, weight, human template, compiled pattern)
_RULES: List[Tuple[str, float, str, re.Pattern]] = [
    ("credential_request", 0.85,
     "It asks for a one-time code, PIN or password. No legitimate bank, company or "
     "government office ever asks for these - that request alone is close to proof of a scam.",
     re.compile(r"\b(otp|one[- ]time (?:code|password|pin)|cvv|pin\s*(?:number|code)?|"
                r"password|passcode|security code|verification code|2fa code|mpin)\b", re.I)),

    ("payment_demand", 0.75,
     "It asks you to send money or buy vouchers. Gift cards, crypto and instant transfers "
     "are the usual choices because they cannot be reversed once sent.",
     re.compile(r"\b(gift\s?card|google\s?play\s?card|itunes\s?card|steam\s?card|bitcoin|"
                r"btc|usdt|crypto|wire\s?transfer|western\s?union|upi|paytm|"
                r"send\s+(?:me\s+)?(?:the\s+)?money|transfer\s+(?:the\s+)?(?:funds|amount)|"
                r"pay\s+(?:a\s+)?(?:fee|fine|charge))\b", re.I)),

    ("urgency_pressure", 0.55,
     "It pushes you to act immediately. Manufactured time pressure is designed to stop you "
     "checking with anyone before you act.",
     re.compile(r"\b(urgent(?:ly)?|immediate(?:ly)?|right now|within \d+\s*(?:hours?|minutes?|days?)|"
                r"before it'?s too late|act now|last (?:chance|warning)|final notice|expires? (?:today|soon))\b", re.I)),

    ("account_threat", 0.65,
     "It threatens to suspend, block or close your account. Real providers notify you through "
     "their own app or website, not with a link in a message.",
     re.compile(r"\b(suspend(?:ed|ing)?|block(?:ed|ing)?|deactivat(?:e|ed|ing)|"
                r"clos(?:e|ed|ing)\s+your\s+account|account\s+(?:has been\s+)?(?:locked|frozen|limited)|"
                r"unusual\s+(?:activity|login)|unauthori[sz]ed\s+access)\b", re.I)),

    ("shortened_link", 0.60,
     "It contains a shortened link, which hides where you would actually be sent.",
     re.compile(r"\b(?:https?://)?(?:bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly|"
                r"rebrand\.ly|cutt\.ly|shorturl\.at|rb\.gy|tiny\.cc)/\S+", re.I)),

    ("raw_ip_link", 0.70,
     "A link points at a bare numeric address instead of a named website. Legitimate services "
     "do not do this.",
     re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\S*", re.I)),

    ("lookalike_domain", 0.65,
     "A web address imitates a well-known brand with altered spelling or an unexpected ending.",
     re.compile(r"\b(?:[a-z0-9-]*(?:payp[a4]l|amaz[o0]n|micr[o0]s[o0]ft|netfl[i1]x|g[o0]{2}gle|"
                r"faceb[o0]{2}k|whatsapp|instagr[a4]m|app1e|apple)[a-z0-9-]*)"
                r"\.(?!com\b|net\b|org\b|co\.uk\b|in\b)[a-z]{2,}\b", re.I)),

    ("punycode_domain", 0.80,
     "A web address uses disguised characters that can make a fake domain look identical to a real one.",
     re.compile(r"\bxn--[a-z0-9-]+\.[a-z]{2,}\b", re.I)),

    ("prize_bait", 0.70,
     "It claims you have won something or are owed money you never applied for.",
     re.compile(r"\b(you(?:'ve| have)? (?:won|been selected)|congratulations|lottery|jackpot|"
                r"prize|lucky winner|claim your (?:reward|prize|refund)|"
                r"(?:tax|customs)\s+refund|unclaimed (?:funds|money)|inheritance)\b", re.I)),

    ("authority_impersonation", 0.55,
     "It claims to be from a bank, tax office, courier or police force - the identities scammers "
     "borrow most, because people react to them quickly.",
     re.compile(r"\b(income\s?tax|tax\s+department|irs\b|hmrc|customs|police|court|legal action|"
                r"cyber\s?crime|bank\s+of\b|federal|social security|medicare|"
                r"(?:fedex|dhl|ups|royal mail|india post)\b.{0,40}\b(?:parcel|package|delivery|customs))\b", re.I)),

    ("channel_switch", 0.45,
     "It tries to move you to a private chat app, away from a place where the conversation could "
     "be checked or reported.",
     re.compile(r"\b(?:message|contact|reach|text|dm|whats\s?app|telegram|signal)\s+(?:me|us)\s+"
                r"(?:on|at|via)\s+(?:whats\s?app|telegram|signal|\+?\d)", re.I)),
]


class TextDetector(Detector):
    modality = Modality.TEXT

    def _run(self, data: Any, meta: Dict[str, Any]) -> Evidence:
        text = data.decode("utf-8", errors="ignore") if isinstance(data, (bytes, bytearray)) else str(data or "")
        text = text.strip()

        signals: List[Signal] = []
        reliability = Reliability()
        contributions: List[float] = []
        findings: Dict[str, Any] = {"characters": len(text), "words": len(text.split())}

        if len(text) < 8:
            return self.abstain(
                "too_short",
                "There is not enough text here to analyse.",
                flag="text_too_short", extra=findings)

        # -- 1. rules --------------------------------------------------------
        hits: List[Dict[str, Any]] = []
        for name, weight, human, pattern in _RULES:
            found = list(pattern.finditer(text))
            if not found:
                continue
            quoted = [m.group(0)[:60] for m in found[:3]]
            line = text[:found[0].start()].count("\n") + 1
            hits.append({"rule": name, "matches": quoted, "count": len(found)})
            contributions.append(weight * 0.55)
            signals.append(Signal(
                name=name, direction=Direction.MANIPULATION,
                human=f"{human} Found: " + ", ".join(f'"{q}"' for q in quoted),
                value={"matches": quoted, "count": len(found)},
                weight=weight, where=f"line {line}"))

        findings["rule_hits"] = hits

        # Several independent red flags together are far stronger than any one.
        if len(hits) >= 3:
            contributions.append(0.35)
            signals.append(Signal(
                name="multiple_independent_red_flags", direction=Direction.MANIPULATION,
                human=(f"{len(hits)} separate warning signs appear in the same short message. "
                       f"Genuine messages rarely trip more than one."),
                value=[h["rule"] for h in hits], weight=0.80, where="whole message"))

        if not hits:
            signals.append(Signal(
                name="no_scam_patterns", direction=Direction.AUTHENTIC,
                human="None of the common scam patterns appear in this message.",
                value=0, weight=0.25, where="whole message"))

        # -- 2. language-model classification --------------------------------
        available, reason = llm.available()
        findings["llm"] = {"available": available, "reason": reason}

        if available:
            verdict = self._classify(text, hits)
            findings["llm_classification"] = verdict
            if verdict:
                category = str(verdict.get("category", "")).lower()
                confidence = float(verdict.get("confidence", 0) or 0)
                reasons = [str(r)[:160] for r in (verdict.get("reasons") or [])][:3]
                if category in ("scam", "phishing", "misinformation") and confidence > 0.4:
                    contributions.append(min(0.55, confidence * 0.6))
                    signals.append(Signal(
                        name=f"llm_{category}", direction=Direction.MANIPULATION,
                        human=(f"A language model reading the whole message classified it as "
                               f"{category}. " + (" ".join(reasons) if reasons else "")),
                        value={"category": category, "confidence": round(confidence, 2)},
                        weight=min(1.0, 0.4 + confidence * 0.5), where="whole message"))
                elif category == "safe":
                    signals.append(Signal(
                        name="llm_safe", direction=Direction.AUTHENTIC,
                        human="A language model reading the whole message found nothing suspicious in it.",
                        value={"category": category, "confidence": round(confidence, 2)},
                        weight=0.30, where="whole message"))

            # -- 3. SelfCheck on the central factual claim -------------------
            # Skipped when the message is already plainly a scam. SelfCheck
            # costs three more API calls, and asking whether a phishing SMS is
            # "factually supported" tells us nothing we do not already know.
            # On the free Gemini tier those wasted calls are what trips the
            # rate limit for the next real analysis.
            already_obvious = len(hits) >= 2
            findings["selfcheck_skipped"] = already_obvious
            consistency = None if already_obvious else self._selfcheck(text)
            if consistency:
                findings["selfcheck"] = consistency
                if consistency["divergence"] >= 0.5:
                    contributions.append(0.30)
                    signals.append(Signal(
                        name="claim_not_consistently_supported", direction=Direction.MANIPULATION,
                        human=(f"We asked a language model about the main factual claim here "
                               f"{consistency['samples']} separate times and got conflicting "
                               f"answers. Claims that are well established produce the same "
                               f"answer every time; this one did not."),
                        value=consistency, weight=0.55, where="main claim"))
                elif consistency["divergence"] <= 0.2 and consistency.get("verdict") == "supported":
                    signals.append(Signal(
                        name="claim_consistently_supported", direction=Direction.AUTHENTIC,
                        human=("The main factual claim here got the same answer every time we "
                               "asked, which is what well-established information looks like."),
                        value=consistency, weight=0.25, where="main claim"))
        else:
            reliability.soft_flags.append("no_language_model")
            signals.append(Signal(
                name="language_model_unavailable", direction=Direction.LIMITATION,
                human=("No language model is available, so this message was checked against "
                       "known scam patterns only. A well-written scam using none of those "
                       f"patterns could slip through. (reason: {reason})"),
                value=reason, weight=0.30, where="whole message"))

        score = schemas.combine(contributions)
        findings["contributions"] = [round(c, 3) for c in contributions]
        return self.build(self.modality, score, signals, reliability, {}, findings)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _classify(text: str, hits: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        prompt = f"""You are a fraud analyst. Classify the message below.

Rule-based scanning already flagged these patterns (may be empty):
{[h['rule'] for h in hits]}

Judge the message itself. Do not assume it is a scam because patterns were
flagged, and do not assume it is safe because none were. Ordinary messages
about deliveries, payments or accounts are common and usually legitimate.

MESSAGE:
\"\"\"{text[:4000]}\"\"\"

Return STRICT JSON only:
{{"category":"scam|phishing|misinformation|safe","confidence":0.0-1.0,"reasons":["short phrase","..."]}}"""
        return llm.generate_json(prompt, model_name=config.TEXT_MODEL, temperature=0.1)

    @staticmethod
    def _selfcheck(text: str) -> Dict[str, Any] | None:
        """
        SelfCheckGPT, applied to the message's central factual claim.

        We ask the same question several times at high temperature. If the
        answers contradict each other, the model has no settled knowledge of
        the claim - which is the signal we want for misinformation.
        """
        extract = llm.generate_json(
            f"""Identify the single most important CHECKABLE factual claim in this message -
something that is either true or false about the world. Ignore instructions,
greetings, requests and opinions.

If there is no checkable factual claim, return {{"claim": null}}.

MESSAGE:
\"\"\"{text[:2000]}\"\"\"

Return STRICT JSON: {{"claim":"<the claim, or null>"}}""",
            model_name=config.TEXT_MODEL, temperature=0.0)

        claim = (extract or {}).get("claim")
        if not claim or not str(claim).strip() or str(claim).lower() == "null":
            return None
        claim = str(claim)[:400]

        samples = llm.sample_json(
            f"""Is the following claim well supported by established knowledge?

CLAIM: "{claim}"

Answer honestly; "unsupported" and "unknown" are valid answers.
Return STRICT JSON: {{"verdict":"supported|unsupported|unknown","why":"one short clause"}}""",
            n=config.VLM_SAMPLES, model_name=config.TEXT_MODEL, temperature=1.0)

        if len(samples) < 2:
            return None

        verdicts = [str(s.get("verdict", "unknown")).lower() for s in samples]
        top = max(set(verdicts), key=verdicts.count)
        agreement = verdicts.count(top) / len(verdicts)
        return {
            "claim": claim,
            "samples": len(samples),
            "verdicts": verdicts,
            "verdict": top,
            "agreement": round(agreement, 2),
            "divergence": round(1.0 - agreement, 2),
        }
