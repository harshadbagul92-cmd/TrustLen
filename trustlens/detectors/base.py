"""Common base for every detection lane."""
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..schemas import (Direction, Evidence, Modality, Reliability, Signal,
                       Verdict, decide)


class Detector(ABC):
    """
    A lane takes raw input and returns Evidence. That is the whole interface.

    Subclasses implement `_run`. The public `analyse` wraps it so that a crash
    inside a lane becomes an honest abstention rather than a 500 or, worse, a
    confident-looking default score.
    """
    modality: Modality

    @abstractmethod
    def _run(self, data: Any, meta: Dict[str, Any]) -> Evidence:
        ...

    def analyse(self, data: Any, meta: Optional[Dict[str, Any]] = None) -> Evidence:
        started = time.perf_counter()
        try:
            evidence = self._run(data, meta or {})
        except Exception as exc:
            evidence = self.abstain(
                f"{type(exc).__name__}",
                "Something went wrong while analysing this file, so no result "
                "is being reported. A failed check is not the same as a clean one.",
                flag="analysis_crashed",
            )
        evidence.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return evidence

    # -- helpers shared by every lane -------------------------------------
    def abstain(self, detail: str, human: str, *, flag: str = "cannot_analyse",
                extra: Optional[Dict[str, Any]] = None) -> Evidence:
        """
        Refuse to judge, and say why.

        Used whenever a check could not run. This is deliberately NOT a low
        score: "we found nothing" and "we could not look" must never render
        the same way.
        """
        from .. import config
        reliability = Reliability(ood_flags=[flag], band=0.25, note=human)
        provenance: Dict[str, Any] = {}
        verdict, confidence = decide(config.ABSTAIN_SCORE, reliability, provenance)
        return Evidence(
            modality=self.modality,
            verdict=verdict,
            score=config.ABSTAIN_SCORE,
            confidence=confidence,
            signals=[Signal(name=flag, direction=Direction.LIMITATION,
                            human=human, value=detail, weight=0.0)],
            reliability=reliability,
            provenance=provenance,
            findings=dict(extra or {}, error=detail),
        )

    @staticmethod
    def build(modality: Modality, score: float, signals: List[Signal],
              reliability: Reliability, provenance: Dict[str, Any],
              findings: Dict[str, Any]) -> Evidence:
        if reliability.soft_flags and not reliability.note:
            reliability.note = "Reduced confidence: " + ", ".join(reliability.soft_flags)
        verdict, confidence = decide(score, reliability, provenance)
        return Evidence(modality=modality, verdict=verdict, score=score,
                        confidence=confidence, signals=signals,
                        reliability=reliability, provenance=provenance,
                        findings=findings)
