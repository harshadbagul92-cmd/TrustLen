from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PredictionEnum(str, Enum):
    LIKELY_AUTHENTIC = "likely_authentic"
    LIKELY_SYNTHETIC = "likely_synthetic"
    LIKELY_MANIPULATED = "likely_manipulated"
    DECLARED_AI = "declared_ai"
    UNCERTAIN = "uncertain"


class ConfidenceLevelEnum(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EvidenceCategoryEnum(str, Enum):
    AI_DETECTOR_SIGNAL = "detector_signal"
    PROVENANCE_INDICATOR = "provenance_indicator"
    EDITING_METADATA = "editing_metadata"
    UNKNOWN_PROVENANCE = "unknown_provenance"
    COMPRESSION_UNCERTAINTY = "compression_uncertainty"
    FACIAL_TEXTURE_ANOMALY = "facial_texture_anomaly"
    BLENDING_ARTIFACT = "blending_artifact"
    LIGHTING_INCONSISTENCY = "lighting_inconsistency"
    SHADOW_REFLECTION_INCONSISTENCY = "shadow_reflection_inconsistency"
    GEOMETRY_ANOMALY = "geometry_anomaly"
    BACKGROUND_ARTIFACT = "background_artifact"
    OBJECT_BOUNDARY_ARTIFACT = "object_boundary_artifact"
    METADATA_SIGNAL = "metadata_signal"
    PROVENANCE_SIGNAL = "provenance_signal"
    DETECTOR_DISAGREEMENT = "detector_disagreement"
    AUTHENTIC_CAMERA_SIGNAL = "authentic_camera_signal"


class EvidenceItem(BaseModel):
    category: EvidenceCategoryEnum
    description: str
    score: float = Field(..., ge=0.0, le=1.0)
    location: Optional[Any] = None


class VisualDetectorOutput(BaseModel):
    model: str
    task: str = "whole_image_synthesis"  # "whole_image_synthesis" | "facial_manipulation_deepfake"
    raw_label: str
    scores: Dict[str, float] = Field(default_factory=dict)
    detector_score: float = Field(..., ge=0.0, le=1.0)
    signal: str = "unknown"  # "synthetic" | "manipulation" | "authentic" | "unknown"
    reliability: str = "medium"  # "high" | "medium" | "low"
    limitations: List[str] = Field(default_factory=list)


class ProvenanceInfo(BaseModel):
    c2pa_present: bool = False
    c2pa_ai_declared: bool = False
    c2pa_edit_declared: bool = False
    software: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    exif_found: bool = False
    exif_tags: Dict[str, Any] = Field(default_factory=dict)

    # Backward compatibility properties
    @property
    def metadata_found(self) -> bool:
        return self.exif_found or self.software is not None or self.camera_model is not None or bool(self.exif_tags)

    @property
    def c2pa_found(self) -> bool:
        return self.c2pa_present or self.c2pa_ai_declared

    @property
    def editing_software(self) -> Optional[str]:
        return self.software


class RegionInfo(BaseModel):
    region_type: str = Field(..., description="Region type e.g. face, eyes, mouth, background")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0)


class ModelMeta(BaseModel):
    name: str
    provider: str
    raw_labels: List[str] = Field(default_factory=list)
    raw_scores: List[float] = Field(default_factory=list)


class DetectorResult(BaseModel):
    modality: str = "image"
    prediction: PredictionEnum
    detector_score: float = Field(..., ge=0.0, le=1.0)
    decision_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevelEnum
    uncertainty: float = Field(..., ge=0.0, le=1.0)
    uncertainty_reason: Optional[str] = None
    decision_rules_triggered: List[str] = Field(default_factory=list)
    detector_outputs: List[VisualDetectorOutput] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    provenance: ProvenanceInfo = Field(default_factory=ProvenanceInfo)
    regions: List[RegionInfo] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    model: ModelMeta
    explanation: Optional[str] = ""


class APIResponse(BaseModel):
    success: bool = True
    result: DetectorResult

