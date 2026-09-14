"""
The shared explainer: one Evidence object in, plain English out.

Two guarantees, both enforced here rather than trusted to the model:

  1. GROUNDING. The prompt carries only the signals present in the Evidence,
     and forbids adding anything else. Whatever the model returns is then
     checked against those signals; if the check fails we ship the template
     instead. A wrong explanation attached to a correct verdict is worse than
     a plain one, because it teaches the reader to look for the wrong thing.

  2. SCOPE. If any LIMITATION signals are present, the explanation must say
     what was NOT checked. "No evidence of manipulation" means only that the
     checks which ran found nothing.

With no API key the template path runs, which reads from exactly the same
signals. The app is fully usable without a model.
"""
import re
from typing import List, Optional

from . import config, llm
from .schemas import Direction, Evidence, Verdict

_PROMPT = """You are TrustLens, explaining a media-authenticity result to someone with no
technical background.

STRICT RULES
1. Use ONLY the findings listed below. Never add a detail, cause or technique
   that is not there.
2. Never name a manipulation method (face swap, GAN, diffusion, voice clone)
   unless a finding names it. Describe what was measured.
3. For "likely_authentic", say "no evidence of manipulation was found" - never
   "this is real" or "this is authentic".
4. If limitations are listed, you MUST state what was not checked.
5. Plain words. No jargon. 3 to 5 sentences. No bullet points, no headings.
6. Do not invent numbers. Quote only figures that appear below.
7. Start with the finding itself. Never introduce yourself, never say "I am
   TrustLens", "I have analysed" or "the result is" - the reader can already
   see the verdict and the score on screen. Write about the content, not about
   the process of checking it.

RESULT
Type: {modality}
Verdict: {verdict}
Suspicion score: {score} (plausible range {low} to {high})

WHAT POINTS TO MANIPULATION:
{manipulation}

WHAT POINTS TO IT BEING GENUINE:
{authentic}

WHAT COULD NOT BE CHECKED:
{limitations}

OTHER CONTEXT:
{context}

Write the explanation now."""

_BANNED = re.compile(
    r"\b(face[- ]?swap|deepfake|gan\b|diffusion|stable diffusion|midjourney|"
    r"voice clon|vocoder|neural network|lip[- ]?sync)\b", re.I)


def _bullets(signals) -> str:
    if not signals:
        return "  (none)"
    return "\n".join(f"  - {s.human}" + (f" [{s.where}]" if s.where else "")
                     for s in signals)


def _template(evidence: Evidence) -> str:
    """Deterministic fallback. Reads the same signals the model would."""
    manip = evidence.by_direction(Direction.MANIPULATION)
    auth = evidence.by_direction(Direction.AUTHENTIC)
    limits = evidence.limitations
    noun = evidence.modality.value

    if evidence.verdict == Verdict.DECLARED_AI:
        head = (f"This {noun} carries metadata in which the file itself states it was "
                f"created by AI. That is a declaration from the file, not a guess.")
        body = " ".join(s.human for s in manip[:2])
        return f"{head} {body}".strip()

    if evidence.verdict == Verdict.LIKELY_AUTHENTIC:
        head = f"No evidence of manipulation was found in this {noun}."
        body = " ".join(s.human for s in auth[:2]) or "None of the checks that ran found anything unusual."
        tail = (" Worth knowing what was not checked: " + " ".join(s.human for s in limits[:2])) if limits else ""
        return f"{head} {body}{tail}"

    if evidence.verdict == Verdict.LIKELY_MANIPULATED:
        head = f"This {noun} shows signs of manipulation."
        body = " ".join(s.human for s in manip[:3])
        tail = (" What was not checked: " + " ".join(s.human for s in limits[:1])) if limits else ""
        return f"{head} {body}{tail}"

    # UNCERTAIN
    reasons = []
    if evidence.reliability.ood_flags:
        reasons.append("the analysis could not run properly")
    if config.AUTHENTIC_BELOW <= evidence.score <= config.MANIPULATED_ABOVE:
        reasons.append("the result sits in the middle, where it could go either way")
    if evidence.reliability.soft_flags:
        reasons.append("conditions that reduce how much the checks can tell us")
    head = (f"We cannot give a confident answer for this {noun}, because "
            f"{' and '.join(reasons) if reasons else 'the signals disagree'}. "
            f"The true score could plausibly be anywhere between "
            f"{evidence.confidence[0]:.2f} and {evidence.confidence[1]:.2f}.")
    seen = (" What we did see: " + " ".join(s.human for s in manip[:2])) if manip else ""
    tail = (" What we could not check: " + " ".join(s.human for s in limits[:2])) if limits else ""
    return f"{head}{seen}{tail}"


def _grounded(text: str, evidence: Evidence) -> bool:
    """
    Reject an explanation that names a technique no signal mentioned.

    Cheap, but it catches the failure that matters: a model dressing up a
    weak compression finding as "a face swap was detected".
    """
    if not text or len(text) < 40:
        return False
    corpus = " ".join(s.human + " " + str(s.value) for s in evidence.signals).lower()
    for match in _BANNED.finditer(text):
        term = match.group(0).lower()
        if term not in corpus:
            return False
    return True


def explain(evidence: Evidence, *, use_model: bool = True) -> str:
    """Fill evidence.explanation. Always returns something usable."""
    fallback = _template(evidence)

    available, _ = llm.available()
    if not use_model or not available:
        return fallback

    prompt = _PROMPT.format(
        modality=evidence.modality.value,
        verdict=evidence.verdict.value,
        score=f"{evidence.score:.2f}",
        low=f"{evidence.confidence[0]:.2f}",
        high=f"{evidence.confidence[1]:.2f}",
        manipulation=_bullets(evidence.by_direction(Direction.MANIPULATION)),
        authentic=_bullets(evidence.by_direction(Direction.AUTHENTIC)),
        limitations=_bullets(evidence.limitations),
        context=_bullets(evidence.by_direction(Direction.CONTEXT)),
    )

    text = llm.generate(prompt, model_name=config.TEXT_MODEL,
                        temperature=0.2, max_tokens=400)
    if text and _grounded(text, evidence):
        return text.strip()
    return fallback
