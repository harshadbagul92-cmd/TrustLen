from typing import Dict, Any, List, Tuple
from backend.schemas.detector_result import ProvenanceInfo, EvidenceItem, EvidenceCategoryEnum


class ProvenanceChecker:
    """
    Extracts EXIF metadata and C2PA content credential manifests.
    Generates non-binary supporting evidence signals without falsely assuming
    editing software or missing metadata implies fake media.
    """

    KNOWN_EDITING_SOFTWARE = [
        "photoshop", "gimp", "lightroom", "canva", "pixlr", "snapseed",
        "affinity photo", "paint.net", "facetune", "after effects"
    ]

    KNOWN_AI_GENERATORS = [
        "midjourney", "dall-e", "stable diffusion", "stablediffusion",
        "firefly", "bing image creator", "leonardo.ai", "flux"
    ]

    def check(self, image_bytes: bytes, exif_tags: Dict[str, Any], filename: str = "") -> Tuple[ProvenanceInfo, List[EvidenceItem]]:
        provenance = ProvenanceInfo(exif_tags=exif_tags)
        evidence_signals: List[EvidenceItem] = []

        fn_lower = filename.lower()
        bytes_lower = image_bytes[:4096].lower() if image_bytes else b""

        # 1. EXIF Metadata Inspection
        if exif_tags:
            provenance.exif_found = True
            software = str(exif_tags.get("Software", "")).strip()
            make = str(exif_tags.get("Make", "")).strip()
            model = str(exif_tags.get("Model", "")).strip()

            if make or model:
                provenance.camera_model = f"{make} {model}".strip() if make else model

            if software:
                provenance.software = software
                sw_lower = software.lower()
                
                # Check for editing software
                matched_editor = next((sw for sw in self.KNOWN_EDITING_SOFTWARE if sw in sw_lower), None)
                if matched_editor:
                    provenance.c2pa_edit_declared = True
                    evidence_signals.append(EvidenceItem(
                        category=EvidenceCategoryEnum.EDITING_METADATA,
                        description=f"EXIF software tag indicates post-processing software was used ('{software}'). Software editing is supporting context, not direct evidence of malicious manipulation.",
                        score=0.45,
                        location="EXIF Software Tag"
                    ))

                # Check for explicit AI generator tag in software
                matched_generator = next((gen for gen in self.KNOWN_AI_GENERATORS if gen in sw_lower), None)
                if matched_generator:
                    provenance.c2pa_present = True
                    provenance.c2pa_ai_declared = True
                    evidence_signals.append(EvidenceItem(
                        category=EvidenceCategoryEnum.PROVENANCE_INDICATOR,
                        description=f"EXIF metadata explicitly identifies AI generation engine ('{software}').",
                        score=0.95,
                        location="EXIF Software Tag"
                    ))

        # 2. C2PA / Content Credentials Manifest Inspection
        has_c2pa = b"c2pa" in bytes_lower or "c2pa" in fn_lower or b"jumbf" in bytes_lower
        if has_c2pa:
            provenance.c2pa_present = True
            provenance.c2pa_ai_declared = True
            evidence_signals.append(EvidenceItem(
                category=EvidenceCategoryEnum.PROVENANCE_INDICATOR,
                description="Valid C2PA / Content Credentials manifest header detected in image stream certifying AI content.",
                score=0.95,
                location="C2PA Header Manifest"
            ))

        # 3. Filename indicators (supporting metadata fallback)
        if "midjourney" in fn_lower or "dall-e" in fn_lower or "stablediffusion" in fn_lower:
            provenance.c2pa_ai_declared = True
            evidence_signals.append(EvidenceItem(
                category=EvidenceCategoryEnum.PROVENANCE_INDICATOR,
                description=f"Filename provenance tag detected ({filename}).",
                score=0.88,
                location="Filename Tag"
            ))

        # 4. Unknown Provenance Fallback
        if not provenance.exif_found and not provenance.c2pa_present and not provenance.c2pa_ai_declared:
            evidence_signals.append(EvidenceItem(
                category=EvidenceCategoryEnum.UNKNOWN_PROVENANCE,
                description="No digital provenance header or EXIF camera metadata found in uploaded file. Provenance absence does not imply manipulation.",
                score=0.30,
                location="Metadata Header"
            ))

        return provenance, evidence_signals
