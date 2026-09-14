from typing import List, Tuple, Optional
from backend.config import settings
from backend.schemas.detector_result import (
    PredictionEnum, ConfidenceLevelEnum, EvidenceItem, EvidenceCategoryEnum, ProvenanceInfo
)


class ConfidenceEngine:
    """
    Evaluates evidence signals, reliability gates, and model probability to compute
    application-level confidence (0.0 to 1.0), confidence level (low, moderate, high),
    uncertainty score, and finalized prediction.
    Enforces false-positive gating to prevent genuine photos from being misclassified.
    """

    def __init__(self):
        self.thresh_uncertain = settings.CONFIDENCE_THRESHOLD_UNCERTAIN  # 0.60
        self.thresh_moderate = settings.CONFIDENCE_THRESHOLD_MODERATE   # 0.80

    def evaluate(
        self,
        raw_prediction: PredictionEnum,
        raw_score: float,
        provenance: ProvenanceInfo,
        evidence: List[EvidenceItem]
    ) -> Tuple[PredictionEnum, float, ConfidenceLevelEnum, float, Optional[str], List[str], float]:
        """
        Evaluates evidence signals, provenance, and detector scores according to explicit Rules P1 to P6.
        Returns: (final_prediction, confidence, confidence_level, uncertainty, uncertainty_reason, rules_triggered, decision_confidence)
        """
        detector_score = max(0.0, min(1.0, float(raw_score)))
        rules_triggered: List[str] = []

        # Check evidence categories
        has_compression = any(e.category == EvidenceCategoryEnum.COMPRESSION_UNCERTAINTY for e in evidence)
        has_facial_anomaly = any(e.category in [EvidenceCategoryEnum.FACIAL_TEXTURE_ANOMALY, EvidenceCategoryEnum.BLENDING_ARTIFACT] for e in evidence)
        has_provenance_ai = provenance.c2pa_ai_declared or provenance.c2pa_present

        # RULE P1 — STRONG AI PROVENANCE
        if has_provenance_ai:
            rules_triggered.append("RULE_P1_STRONG_AI_PROVENANCE")
            prediction = PredictionEnum.DECLARED_AI if provenance.c2pa_present else PredictionEnum.LIKELY_SYNTHETIC
            decision_confidence = 0.95
            confidence = 0.95
            confidence_level = ConfidenceLevelEnum.HIGH
            uncertainty = 0.05
            uncertainty_reason = None
            return prediction, confidence, confidence_level, uncertainty, uncertainty_reason, rules_triggered, decision_confidence

        # RULE P5 — OOD / LOW QUALITY
        if has_compression:
            rules_triggered.append("RULE_P5_LOW_QUALITY_OOD")
            prediction = PredictionEnum.UNCERTAIN
            decision_confidence = 0.75  # Confident in decision to abstain due to low quality
            confidence = round(max(0.20, detector_score * 0.50), 3)
            confidence_level = ConfidenceLevelEnum.LOW
            uncertainty = round(1.0 - confidence, 3)
            uncertainty_reason = "Heavy image compression or low resolution degrades pixel grid resolution; result defaulted to uncertain."
            return prediction, confidence, confidence_level, uncertainty, uncertainty_reason, rules_triggered, decision_confidence

        # RULE P3 — STRONG MANIPULATION EVIDENCE
        if has_facial_anomaly and detector_score >= 0.85:
            rules_triggered.append("RULE_P3_STRONG_MANIPULATION_EVIDENCE")
            prediction = PredictionEnum.LIKELY_MANIPULATED
            decision_confidence = round(detector_score, 3)
            confidence = round(detector_score, 3)
            confidence_level = ConfidenceLevelEnum.HIGH
            uncertainty = round(1.0 - confidence, 3)
            uncertainty_reason = None
            return prediction, confidence, confidence_level, uncertainty, uncertainty_reason, rules_triggered, decision_confidence

        # RULE P2 — STRONG VISUAL SYNTHETIC + CORROBORATION
        if raw_prediction == PredictionEnum.LIKELY_SYNTHETIC and detector_score >= 0.88:
            rules_triggered.append("RULE_P2_STRONG_VISUAL_SYNTHETIC")
            prediction = PredictionEnum.LIKELY_SYNTHETIC
            decision_confidence = round(detector_score, 3)
            confidence = round(detector_score, 3)
            confidence_level = ConfidenceLevelEnum.HIGH
            uncertainty = round(1.0 - confidence, 3)
            uncertainty_reason = None
            return prediction, confidence, confidence_level, uncertainty, uncertainty_reason, rules_triggered, decision_confidence

        # RULE P4 — CONFLICT / UNCORROBORATED SINGLE DETECTOR SIGNAL
        if raw_prediction == PredictionEnum.LIKELY_SYNTHETIC and detector_score < 0.88:
            rules_triggered.append("RULE_P4_DETECTOR_CONFLICT_UNCORROBORATED")
            prediction = PredictionEnum.UNCERTAIN
            decision_confidence = 0.70
            confidence = round(detector_score * 0.50, 3)
            confidence_level = ConfidenceLevelEnum.LOW
            uncertainty = round(1.0 - confidence, 3)
            uncertainty_reason = "Single visual detector signal lacks multi-model or provenance corroboration; defaulted to uncertain to protect genuine photos."
            return prediction, confidence, confidence_level, uncertainty, uncertainty_reason, rules_triggered, decision_confidence

        # RULE P6 — SAFE AUTHENTIC
        if raw_prediction == PredictionEnum.LIKELY_AUTHENTIC and detector_score >= self.thresh_uncertain:
            rules_triggered.append("RULE_P6_SAFE_AUTHENTIC")
            prediction = PredictionEnum.LIKELY_AUTHENTIC
            decision_confidence = round(detector_score, 3)
            confidence = round(detector_score, 3)
            confidence_level = ConfidenceLevelEnum.HIGH if detector_score >= self.thresh_moderate else ConfidenceLevelEnum.MODERATE
            uncertainty = round(1.0 - confidence, 3)
            uncertainty_reason = None
            return prediction, confidence, confidence_level, uncertainty, uncertainty_reason, rules_triggered, decision_confidence

        # DEFAULT FALLBACK — UNCERTAIN
        rules_triggered.append("RULE_DEFAULT_UNCERTAIN")
        prediction = PredictionEnum.UNCERTAIN
        decision_confidence = 0.60
        confidence = round(abs(detector_score - 0.50) * 2.0, 3)
        confidence_level = ConfidenceLevelEnum.LOW
        uncertainty = round(1.0 - confidence, 3)
        uncertainty_reason = f"Detector score ({detector_score:.2f}) is inconclusive; defaulted to uncertain."
        return prediction, confidence, confidence_level, uncertainty, uncertainty_reason, rules_triggered, decision_confidence


