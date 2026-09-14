import logging
from backend.detectors.image.preprocessing import ImagePreprocessor
from backend.detectors.image.provenance import ProvenanceChecker
from backend.detectors.image.face_detection import RegionDetector
from backend.detectors.image.hf_image_detector import HuggingFaceImageDetector
from backend.detectors.image.evidence import EvidenceEngine
from backend.services.confidence import ConfidenceEngine
from backend.schemas.detector_result import DetectorResult, ModelMeta, PredictionEnum, EvidenceCategoryEnum

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    """
    Orchestration service coordinating image preprocessing, provenance checking,
    region detection, HF model detection, evidence synthesis, and confidence evaluation.
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.provenance_checker = ProvenanceChecker()
        self.region_detector = RegionDetector()
        self.hf_detector = HuggingFaceImageDetector()
        self.evidence_engine = EvidenceEngine()
        self.confidence_engine = ConfidenceEngine()

    def analyze_image(self, image_bytes: bytes, filename: str = "upload.jpg") -> DetectorResult:
        logger.info(f"[Image Pipeline] Starting analysis for file '{filename}' ({len(image_bytes)} bytes).")

        # Step 1: Preprocessing
        prep_result = self.preprocessor.preprocess(image_bytes, filename=filename)
        logger.info(f"[Image Pipeline] Preprocessing complete. Dimensions: {prep_result.width}x{prep_result.height}, Format: {prep_result.format_name}.")

        # Step 2: Provenance / Metadata Check
        provenance, prov_evidence = self.provenance_checker.check(
            image_bytes=image_bytes,
            exif_tags=prep_result.exif_raw,
            filename=filename
        )

        # Step 3: Region / Face Detection
        regions, region_evidence = self.region_detector.detect_regions(prep_result.normalized_image)

        # Step 4: Hugging Face Detector Adapter Inference
        hf_output = self.hf_detector.detect(prep_result.normalized_image, filename=filename)

        # Step 5: Evidence Engine Synthesis
        evidence_list, warnings = self.evidence_engine.synthesize(
            hf_output=hf_output,
            provenance=provenance,
            prov_evidence=prov_evidence,
            regions=regions,
            region_evidence=region_evidence,
            image_format=prep_result.format_name,
            size_bytes=prep_result.size_bytes
        )

        # Step 6: Confidence & Decision Engine Evaluation
        final_prediction, confidence, confidence_level, uncertainty, uncertainty_reason, rules_triggered, decision_confidence = self.confidence_engine.evaluate(
            raw_prediction=hf_output.prediction,
            raw_score=hf_output.raw_score,
            provenance=provenance,
            evidence=evidence_list
        )

        # Step 7: Explanation Layer Synthesis
        explanation_text = self._build_explanation(
            prediction=final_prediction,
            detector_score=hf_output.raw_score,
            confidence_level=confidence_level,
            evidence=evidence_list,
            provenance=provenance
        )

        # Step 8: Construct Structured DetectorResult
        result = DetectorResult(
            modality="image",
            prediction=final_prediction,
            detector_score=round(hf_output.raw_score, 4),
            decision_confidence=round(decision_confidence, 4),
            confidence=confidence,
            confidence_level=confidence_level,
            uncertainty=uncertainty,
            uncertainty_reason=uncertainty_reason,
            decision_rules_triggered=rules_triggered,
            detector_outputs=[hf_output.to_visual_output()],
            evidence=evidence_list,
            provenance=provenance,
            regions=regions,
            warnings=warnings,
            model=ModelMeta(
                name=hf_output.model_name,
                provider=hf_output.provider,
                raw_labels=hf_output.raw_labels,
                raw_scores=hf_output.raw_scores
            ),
            explanation=explanation_text
        )

        logger.info(f"[Image Pipeline] Analysis complete. Result: {result.prediction} (Detector Score: {result.detector_score:.4f}, Decision Confidence: {result.decision_confidence:.2f}).")
        return result

    def _build_explanation(
        self,
        prediction: PredictionEnum,
        detector_score: float,
        confidence_level: str,
        evidence: list,
        provenance: any = None
    ) -> str:
        """Builds plain-English grounded narrative strictly based on observed evidence."""
        if prediction == PredictionEnum.DECLARED_AI:
            return (
                "Content credentials and provenance metadata explicitly certify that this image was generated "
                "or edited using an AI software system. This assessment is based on verified metadata headers."
            )
        elif prediction == PredictionEnum.LIKELY_SYNTHETIC:
            return (
                f"The image detector identified visual feature patterns associated with whole-image AI generation "
                f"(detector score: {detector_score:.4f}). High detector confidence and consistent signals support a likely synthetic assessment."
            )
        elif prediction == PredictionEnum.LIKELY_MANIPULATED:
            return (
                f"Region and facial analysis identified structural anomalies or localized blending discontinuities "
                f"(detector score: {detector_score:.4f}). Evidence indicates localized digital image manipulation."
            )
        elif prediction == PredictionEnum.LIKELY_AUTHENTIC:
            return (
                f"No significant evidence of AI generation or digital manipulation was found across inspected parameters "
                f"(detector score: {detector_score:.4f}). The image appears consistent with camera photography. Note: this assessment does not guarantee 100% authenticity."
            )
        else: # UNCERTAIN
            if any(e.category == EvidenceCategoryEnum.COMPRESSION_UNCERTAINTY for e in evidence):
                return (
                    "The image resolution is very small or heavily compressed, which degrades pixel grid analysis. "
                    "Because high-frequency forensic signals are obscured, the system avoids making a confident authenticity claim."
                )
            return (
                f"Visual detector signals lack multi-model corroboration or conflict with available evidence "
                f"(detector score: {detector_score:.4f}). Because the signals are inconclusive, the result is defaulted to uncertain to protect against false accusations."
            )

