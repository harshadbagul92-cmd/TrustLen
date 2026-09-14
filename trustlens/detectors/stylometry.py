"""
Measurable writing-style properties, used to ask whether text reads as
machine-generated.

Be clear about the ceiling here. Detecting AI-written text is genuinely
unreliable — OpenAI withdrew its own AI Text Classifier in 2023 for low
accuracy, and every academic result since has been fragile to paraphrasing.
Nothing in this module is a detector. These are real, checkable properties of
a piece of writing that correlate loosely with machine generation, and the
lane weights them as the weak evidence they are.

The strongest of them is burstiness. Human writing varies its sentence length
a lot — a three-word sentence next to a forty-word one. Model output tends to
settle into a narrower band. It is a tendency, not a rule: a press release
written by a person is also very even.
"""
import re
from collections import Counter
from typing import Any, Dict, List

# Words and phrases that became markedly more common in assistant-written prose.
# Presence proves nothing on its own — people write these too — so this only
# ever contributes a small amount alongside other measurements.
_TELLS = [
    "delve", "delves", "delving", "tapestry", "testament to", "navigate the",
    "in the realm of", "it is important to note", "it's important to note",
    "it is worth noting", "plays a crucial role", "plays a vital role",
    "a beacon of", "underscores the", "multifaceted", "pivotal role",
    "in today's fast-paced", "ever-evolving", "seamlessly integrate",
    "let's dive", "unlock the potential", "when it comes to", "furthermore,",
    "moreover,", "in conclusion,", "on the other hand,", "that being said,",
]

_SENT_SPLIT = re.compile(r"[.!?]+[\s\n]+|\n{2,}")
_WORD = re.compile(r"[A-Za-z']+")
_CONTRACTION = re.compile(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.I)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 1]


def measure(text: str) -> Dict[str, Any]:
    """
    Every number returned is computed from the text in front of us.

    Returns an empty-ish dict for input too short to say anything about;
    the caller must treat a missing key as "not measured", never as zero.
    """
    out: Dict[str, Any] = {}
    words = _WORD.findall(text)
    sents = _sentences(text)
    out["word_count"] = len(words)
    out["sentence_count"] = len(sents)
    if len(words) < 25 or len(sents) < 3:
        out["too_short"] = True
        return out
    out["too_short"] = False

    lengths = [len(_WORD.findall(s)) for s in sents]
    lengths = [n for n in lengths if n > 0]
    if len(lengths) >= 3:
        mean = sum(lengths) / len(lengths)
        var = sum((n - mean) ** 2 for n in lengths) / len(lengths)
        sd = var ** 0.5
        out["mean_sentence_words"] = round(mean, 1)
        # Burstiness: how much sentence length varies, relative to its own mean.
        out["burstiness"] = round(sd / mean, 3) if mean > 0 else 0.0

    lower = [w.lower() for w in words]
    out["lexical_diversity"] = round(len(set(lower)) / len(lower), 3)

    # Repeated three-word phrases: models loop more than people do.
    if len(lower) >= 12:
        tri = Counter(tuple(lower[i:i + 3]) for i in range(len(lower) - 2))
        repeats = sum(c - 1 for c in tri.values() if c > 1)
        out["repeated_trigram_rate"] = round(repeats / max(1, len(tri)), 3)

    out["contraction_rate"] = round(len(_CONTRACTION.findall(text)) / len(words), 4)

    low = text.lower()
    found = [t for t in _TELLS if t in low]
    out["stock_phrases"] = found[:6]
    out["stock_phrase_count"] = len(found)

    # Typos and irregular spacing are weak evidence of a human at a keyboard.
    out["double_spaces"] = text.count("  ")
    out["exclamations"] = text.count("!")
    out["ellipses"] = text.count("...")
    return out


def reads_as_machine(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn the measurements into a small, explicitly weak score.

    Capped deliberately low: style alone should never be enough to call a piece
    of writing machine-generated. It can only nudge, and it says so.
    """
    if m.get("too_short", True):
        return {"score": 0.0, "reasons": [], "measured": False}

    reasons: List[Dict[str, Any]] = []
    score = 0.0

    b = m.get("burstiness")
    if b is not None and b < 0.35:
        score += 0.20
        reasons.append({
            "what": "very even sentence lengths",
            "detail": f"sentence length barely varies (burstiness {b}). People usually mix "
                      f"short and long sentences; generated text often settles into one rhythm.",
            "value": b})

    r = m.get("repeated_trigram_rate")
    if r is not None and r > 0.05:
        score += 0.12
        reasons.append({
            "what": "repeated phrasing",
            "detail": f"{r * 100:.0f}% of three-word phrases repeat, which is more looping than "
                      f"most people do in a short piece.",
            "value": r})

    n = m.get("stock_phrase_count", 0)
    if n >= 2:
        score += min(0.18, 0.06 * n)
        reasons.append({
            "what": "stock assistant phrasing",
            "detail": f"uses {n} phrases that became common in assistant-written prose: "
                      + ", ".join(f'"{p}"' for p in m.get("stock_phrases", [])[:3]),
            "value": m.get("stock_phrases")})

    c = m.get("contraction_rate")
    if c is not None and c == 0 and m.get("word_count", 0) > 60:
        score += 0.08
        reasons.append({
            "what": "no contractions at all",
            "detail": "not a single contraction in a fairly long passage, which is more formal "
                      "than most people write.",
            "value": 0})

    return {"score": round(min(0.45, score), 3), "reasons": reasons, "measured": True}
