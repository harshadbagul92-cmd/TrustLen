from typing import Any, Optional, Dict
from detectors.base import BaseDetector
from detectors.text_rules import TextRuleExtractor
from detectors.selfcheckgpt import SelfCheckGPTScorer
from detectors.text_classifier import LLMTextClassifier
from schemas import (
    Evidence, Signal, Reliability, ModalityEnum, derive_verdict_and_confidence
)


class TextDetector(BaseDetector):
    """
    Member A - Text Detection Lane.
    Combines rule extraction, zero-shot LLM classification, and SelfCheckGPT factual claim evaluation.
    """

    def __init__(self):
        self.rule_extractor = TextRuleExtractor()
        self.selfcheck_scorer = SelfCheckGPTScorer()
        self.llm_classifier = LLMTextClassifier()

    def analyse(self, input_data: Any, metadata: Optional[Dict[str, Any]] = None) -> Evidence:
        text = str(input_data).strip()
        metadata = metadata or {}
        headers = metadata.get("headers", {})
        api_key = metadata.get("api_key")

        reliability = Reliability(ood_flags=[], band=0.0, note="Text lane evaluation")
        provenance = {}

        # Check declared AI provenance tags
        if "[ChatGPT]" in text or "[AI-Generated]" in text or metadata.get("declared_ai"):
            provenance["declared_ai"] = True
            provenance["generator"] = "ChatGPT / LLM Header Tag"

        # 1. Rule Extractor
        rule_signals = self.rule_extractor.extract_signals(text, headers=headers)

        # 2. Zero-Shot LLM Classifier
        classification = self.llm_classifier.classify(text, rule_hits=rule_signals, api_key=api_key)
        cat = classification.get("category", "safe")
        cat_conf = float(classification.get("confidence", 0.15))

        # 3. SelfCheckGPT Factual Claim Scorer
        divergence_score, selfcheck_signal = self.selfcheck_scorer.evaluate_claim(text, api_key=api_key)

        # Assemble all signals
        all_signals = list(rule_signals)
        if selfcheck_signal:
            all_signals.append(selfcheck_signal)

        for rs in classification.get("reasoning_signals", []):
            all_signals.append(Signal(
                name=rs.get("name", "llm_finding"),
                human=rs.get("human", "LLM classifier finding"),
                value=f"Category: {cat}",
                weight=float(rs.get("weight", cat_conf)),
                where=rs.get("where", "Full Document")
            ))

        # OOD Check: Very short input (< 5 words)
        words = text.split()
        if len(words) < 5:
            reliability.ood_flags.append("short_input_length")
            reliability.band = 0.25
            reliability.note = "Input contains fewer than 5 words; reliable perplexity/semantic analysis degraded."

        # Calibrated Score Calculation
        if provenance.get("declared_ai"):
            score = 0.95
        elif cat in ["scam", "phishing"]:
            max_rule_w = max([s.weight for s in rule_signals], default=0.70)
            score = max(cat_conf, max_rule_w)
        elif cat == "misinformation":
            score = max(cat_conf, divergence_score if divergence_score > 0 else 0.75)
        else: # safe
            if rule_signals:
                score = max([s.weight for s in rule_signals]) * 0.7
            else:
                score = 0.10

        score = max(0.0, min(1.0, round(score, 2)))

        verdict, confidence = derive_verdict_and_confidence(score, reliability, provenance)

        return Evidence(
            modality=ModalityEnum.TEXT,
            verdict=verdict,
            score=score,
            confidence=confidence,
            signals=all_signals,
            reliability=reliability,
            provenance=provenance,
            findings={
                "category": cat,
                "category_confidence": cat_conf,
                "selfcheckgpt_divergence": divergence_score,
                "rule_count": len(rule_signals)
            }
        )
