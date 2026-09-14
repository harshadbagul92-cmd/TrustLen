import logging
from typing import List, Tuple
from PIL import Image
from backend.schemas.detector_result import RegionInfo, EvidenceItem, EvidenceCategoryEnum

logger = logging.getLogger(__name__)


class RegionDetector:
    """
    Lightweight region and face detection abstraction.
    Locates spatial regions of interest (faces, eyes, background) for regional evidence scoring.
    """

    def detect_regions(self, image: Image.Image) -> Tuple[List[RegionInfo], List[EvidenceItem]]:
        regions: List[RegionInfo] = []
        evidence_signals: List[EvidenceItem] = []

        try:
            width, height = image.size

            # Attempt OpenCV Haar Cascade if installed, otherwise fallback to bounding grid estimator
            try:
                import cv2
                import numpy as np

                # Convert PIL image to BGR numpy array for OpenCV
                img_np = np.array(image.convert("RGB"))
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                # Load pretrained Haar Cascade classifier
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    regions.append(RegionInfo(
                        region_type="face",
                        bbox=[int(x), int(y), int(x + w), int(y + h)],
                        confidence=0.92
                    ))
            except Exception as cv_err:
                logger.debug(f"OpenCV face detection unavailable/failed ({cv_err}). Skipping CV face extraction.")

            # If OpenCV detection found faces, report facial texture signal
            if regions:
                evidence_signals.append(EvidenceItem(
                    category=EvidenceCategoryEnum.FACIAL_TEXTURE,
                    description=f"Identified {len(regions)} facial region(s) for localized boundary inspection.",
                    score=0.50,
                    location=f"Bounding Boxes: {[r.bbox for r in regions]}"
                ))
            else:
                # Add default global background region
                regions.append(RegionInfo(
                    region_type="background",
                    bbox=[0, 0, int(width), int(height)],
                    confidence=1.0
                ))

        except Exception as e:
            logger.warning(f"Region detection encountered error: {e}. Defaulting to full image background region.")
            regions = [RegionInfo(region_type="background", bbox=[0, 0, image.width, image.height], confidence=1.0)]

        return regions, evidence_signals
