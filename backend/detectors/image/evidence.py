from typing import List, Tuple
from backend.detectors.image.hf_image_detector import DetectorOutput
from backend.schemas.detector_result import (
    EvidenceItem, EvidenceCategoryEnum, ProvenanceInfo, RegionInfo, PredictionEnum
)


class EvidenceEngine:
    """
    Consolidates signals from AI Detector Adapter, Metadata Provenance,
    Region Analysis, and Image Quality into grounded evidence items.
    """

    def synthesize(
        self,
        hf_output: DetectorOutput,
        provenance: ProvenanceInfo,
        prov_evidence: List[EvidenceItem],
        regions: List[RegionInfo],
        region_evidence: List[EvidenceItem],
        image_format: str,
        size_bytes: int
    ) -> Tuple[List[EvidenceItem], List[str]]:
        evidence_list: List[EvidenceItem] = []
        warnings: List[str] = []

        if hf_output.warning:
            warnings.append(hf_output.warning)

        # 1. AI Detector Signal
        if hf_output.prediction == PredictionEnum.LIKELY_SYNTHETIC:
            description_text = f"The image detector produced a synthetic-image signal (detector_score={hf_output.raw_score:.4f}, raw label='{hf_output.raw_label}')."
        elif hf_output.prediction == PredictionEnum.LIKELY_AUTHENTIC:
            description_text = f"The image detector produced an authentic-image signal (detector_score={hf_output.raw_score:.4f}, raw label='{hf_output.raw_label}')."
        else:
            description_text = f"The image detector produced an uncertain signal (detector_score={hf_output.raw_score:.4f}, raw label='{hf_output.raw_label}')."

        evidence_list.append(EvidenceItem(
            category=EvidenceCategoryEnum.AI_DETECTOR_SIGNAL,
            description=description_text,
            score=round(hf_output.raw_score, 4),
            location=None
        ))

        # 2. Provenance Evidence Signals
        for item in prov_evidence:
            # Map categories cleanly to supported set
            if item.category in [EvidenceCategoryEnum.PROVENANCE_SIGNAL, EvidenceCategoryEnum.METADATA_SIGNAL]:
                if "software" in item.description.lower():
                    cat = EvidenceCategoryEnum.EDITING_METADATA
                elif "no digital provenance" in item.description.lower():
                    cat = EvidenceCategoryEnum.UNKNOWN_PROVENANCE
                else:
                    cat = EvidenceCategoryEnum.PROVENANCE_INDICATOR
            else:
                cat = item.category
            
            evidence_list.append(EvidenceItem(
                category=cat,
                description=item.description,
                score=item.score,
                location=item.location
            ))

        # 3. Regional / Facial Evidence Signals
        evidence_list.extend(region_evidence)

        # 4. Image Quality / Compression Uncertainty Check
        if size_bytes < 15 * 1024:  # Extremely small file < 15 KB
            evidence_list.append(EvidenceItem(
                category=EvidenceCategoryEnum.COMPRESSION_UNCERTAINTY,
                description="Extremely small file size or high compression ratio detected; pixel artifacts may obscure model resolution.",
                score=0.40,
                location="Global Pixel Stream"
            ))
            warnings.append("Low file resolution / heavy compression detected.")

        # Important Model Limitation Warning
        warnings.append("Model Scope Note: 'umm-maybe/AI-image-detector' is a global whole-image generator detector. It is not a localized facial deepfake/swap detector.")
        warnings.append("Raw model detector_score is model-dependent and not independently calibrated.")

        return evidence_list, warnings

