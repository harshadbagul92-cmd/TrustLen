"""
Vision-model assessment, filtered by self-consistency.

This approximates SIDA (Huang et al. 2024), which uses a large multimodal
model to detect, localize and explain image manipulation in one pass. We
cannot train SIDA in a hackathon, so instead we ask a general vision model
for SPECIFIC OBSERVABLE artifacts and apply the SelfCheckGPT consistency test
to separate what it really sees from what it invents.

The closed vocabulary below is what makes agreement measurable - free-text
answers cannot be compared across samples.

Important honesty note for the UI: a vision model reasons about semantic
implausibility (six fingers, garbled signage, impossible shadows). It does
NOT read pixel-level forensic traces, and it will miss a well-made face swap
of a real person.
"""
from typing import Any, Dict, List

from .. import config, llm

ARTIFACTS = [
    "anatomy_hands_fingers",
    "anatomy_teeth_eyes_ears",
    "garbled_text",
    "inconsistent_lighting",
    "impossible_reflection",
    "melted_or_repeating_background",
    "waxy_or_oversmooth_skin",
    "mismatched_edges_or_blending",
    "physically_impossible_geometry",
]

_PROMPT = """You are a forensic image analyst. Examine this image and report ONLY
specific, concrete visual artifacts you can actually see and point to.

If the image looks like an ordinary photograph, return an empty list. Reporting
nothing is a correct and valuable answer. Do NOT speculate, and do NOT try to be
helpful by finding something.

Use these exact category strings:
{cats}

Return STRICT JSON only, no prose, no code fence:
{{"observations":[{{"artifact":"<category>","region":"<short location>","detail":"<what you actually see, one clause>","severity":<1-5>}}]}}"""


def assess(pil_image) -> Dict[str, Any]:
    """
    Sample the model N times; keep only what a majority of samples repeat.

    Always safe to read. Returns:
      available   False when there is no key or every call failed
      reason      why unavailable
      samples_ok  how many calls returned parseable JSON
      consistent  observations a majority agreed on  -> these score
      discarded   minority observations              -> shown, never scored
      score       0..1 derived from `consistent` only
    """
    out: Dict[str, Any] = {"available": False, "reason": None, "samples_ok": 0,
                           "consistent": [], "discarded": [], "score": 0.0}

    ok, reason = llm.available()
    if not ok:
        out["reason"] = reason
        return out

    # Downscale before upload: keeps calls fast and cheap, and micro-detail is
    # not what we are asking this model about anyway.
    image = pil_image.copy()
    image.thumbnail((1024, 1024))

    prompt = _PROMPT.format(cats="\n".join(f"  - {c}" for c in ARTIFACTS))
    samples = llm.sample_json(prompt, image=image, model_name=config.VLM_MODEL,
                              temperature=1.0, max_tokens=800)

    out["samples_ok"] = len(samples)
    if not samples:
        out["reason"] = "all_calls_failed"
        return out

    out["available"] = True
    grouped = llm.agreement(samples, "observations", "artifact")
    majority = (len(samples) // 2) + 1

    for artifact, info in grouped.items():
        if artifact not in ARTIFACTS:
            continue
        first = info["first"]
        try:
            severity = float(first.get("severity", 3) or 3)
        except (TypeError, ValueError):
            severity = 3.0
        entry = {
            "artifact": artifact,
            "agreement": f"{info['hits']}/{len(samples)}",
            "ratio": info["ratio"],
            "severity": round(max(1.0, min(5.0, severity)), 1),
            "region": str(first.get("region") or "unspecified")[:80],
            "detail": str(first.get("detail") or "")[:200],
        }
        (out["consistent"] if info["hits"] >= majority else out["discarded"]).append(entry)

    # Score from consistent observations only, combined noisy-OR so that one
    # mild artifact cannot by itself condemn an image.
    if out["consistent"]:
        remaining = 1.0
        for e in out["consistent"]:
            remaining *= (1.0 - min(0.85, e["ratio"] * (e["severity"] / 5.0)))
        out["score"] = round(1.0 - remaining, 3)

    out["consistent"].sort(key=lambda e: e["ratio"] * e["severity"], reverse=True)
    return out
