"""
The Evidence contract.

Every detection lane - text, image, audio, video - returns exactly this
object. The explainer and the UI read only this, so they never need to know
which lane produced a result, and a new lane costs nothing to display.

The rule that governs the whole project: every Signal must correspond to a
measurement that was actually taken. If a check could not run, the lane says
so with a LIMITATION signal. It never fills the gap with a plausible number.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from . import config


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class Verdict(str, Enum):
    DECLARED_AI = "declared_ai"            # the file itself says so
    LIKELY_AUTHENTIC = "likely_authentic"  # "no evidence found", never "this is real"
    UNCERTAIN = "uncertain"                # not enough to call it
    LIKELY_MANIPULATED = "likely_manipulated"


class Direction(str, Enum):
    """What a signal argues for. Lets the UI separate findings from context
    instead of presenting every measurement as if it were an accusation."""
    MANIPULATION = "supports_manipulation"
    AUTHENTIC = "supports_authentic"
    CONTEXT = "context"          # measured, but argues neither way
    LIMITATION = "limitation"    # a check that could NOT run


class Signal(BaseModel):
    name: str = Field(..., description="Stable machine identifier")
    direction: Direction = Field(default=Direction.CONTEXT)
    human: str = Field(..., description="One plain-English sentence a non-technical reader understands")
    value: Any = Field(..., description="The measurement itself. Never None for a real finding.")
    weight: float = Field(..., ge=0.0, le=1.0, description="How much this mattered to the score")
    where: Optional[str] = Field(None, description="Image region, seconds range, frame numbers, or line")


class Reliability(BaseModel):
    """
    Two tiers, and the difference matters.

    ood_flags are HARD: any one forces UNCERTAIN, because the analysis could
    not meaningfully run. soft_flags only widen the confidence band - nearly
    every real-world image is compressed, and forcing UNCERTAIN on all of them
    would make the lane useless.
    """
    ood_flags: List[str] = Field(default_factory=list)
    soft_flags: List[str] = Field(default_factory=list)
    band: float = Field(default=0.0, ge=0.0, le=1.0)
    note: Optional[str] = None


class Evidence(BaseModel):
    modality: Modality
    verdict: Verdict
    score: float = Field(..., ge=0.0, le=1.0, description="Calibrated. Higher = more suspicious.")
    confidence: Tuple[float, float]
    signals: List[Signal] = Field(default_factory=list)
    reliability: Reliability = Field(default_factory=Reliability)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    findings: Dict[str, Any] = Field(default_factory=dict, description="Raw measurements, shown in the drawer")
    explanation: str = Field(default="")
    elapsed_ms: int = Field(default=0)

    @field_validator("confidence")
    @classmethod
    def _order_confidence(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        low, high = sorted((round(float(v[0]), 3), round(float(v[1]), 3)))
        return (max(0.0, low), min(1.0, high))

    # -- convenience used by the UI and the explainer ----------------------
    def by_direction(self, direction: Direction) -> List[Signal]:
        return [s for s in self.signals if s.direction == direction]

    @property
    def limitations(self) -> List[Signal]:
        return self.by_direction(Direction.LIMITATION)


def decide(score: float, reliability: Reliability,
           provenance: Dict[str, Any]) -> Tuple[Verdict, Tuple[float, float]]:
    """
    Turn a raw score into a verdict and a confidence band.

    Order matters:
      1. The file declaring its own AI origin beats any measurement.
      2. A hard reliability flag means we could not analyse - abstain.
      3. Otherwise fall through the bands.
    """
    if provenance.get("declared_ai"):
        verdict = Verdict.DECLARED_AI
    elif reliability.ood_flags:
        verdict = Verdict.UNCERTAIN
    elif score < config.AUTHENTIC_BELOW:
        verdict = Verdict.LIKELY_AUTHENTIC
    elif score > config.MANIPULATED_ABOVE:
        verdict = Verdict.LIKELY_MANIPULATED
    else:
        verdict = Verdict.UNCERTAIN

    delta = config.BASE_CONFIDENCE_MARGIN + reliability.band
    return verdict, (max(0.0, score - delta), min(1.0, score + delta))


def combine(contributions: List[float], base: float = None) -> float:
    """
    Noisy-OR: independent pieces of evidence accumulate, but no single one can
    drive the score to certainty, and finding nothing leaves `base`.
    """
    base = config.BASE_SCORE if base is None else base
    remaining = 1.0 - base
    for c in contributions:
        remaining *= (1.0 - max(0.0, min(0.9, float(c))))
    return round(1.0 - remaining, 3)
