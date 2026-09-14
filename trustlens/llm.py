"""
One small wrapper around the Gemini API, used by three callers:
the text classifier, the image vision check, and the explainer.

Design rules:
  - Never raises. A model failure degrades the result; it never breaks a lane.
  - Never logs prompt or response content.
  - The key is read from the environment, never passed through the UI or
    stored anywhere.

SelfCheckGPT (Manakul et al. 2023) underlies `sample_json`: ask the same
question N times and keep only what the model repeats. Divergence between
samples is the signal that the model is guessing, and it is far more reliable
than the model's own stated confidence.
"""
import json
import re
import time
from typing import Any, Dict, List, Optional

from . import config

_BACKEND_ERROR: Optional[str] = None
_RETRY_AFTER: float = 0.0        # monotonic clock; 0 means "fine to call"

# How long to stop calling after each kind of failure. A rate limit is
# temporary and resets on its own, so it must NOT disable the model for the
# rest of the session - an earlier version latched the first error forever,
# which meant one 429 silently turned off every model check until restart.
_COOLDOWNS = {
    "rate_limit": 25.0,
    "auth": float("inf"),        # a bad key will not fix itself
    "other": 8.0,
}


def _classify_error(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "resourceexhausted" in name or "429" in text or "quota" in text or "rate limit" in text:
        return "rate_limit"
    auth_names = ("permissiondenied", "unauthenticated")
    auth_text = ("api key not valid", "401", "403", "permission denied")
    if any(k in name for k in auth_names) or any(k in text for k in auth_text):
        return "auth"
    return "other"


def last_error() -> Optional[str]:
    """The most recent API failure, for the UI and for debugging."""
    return _BACKEND_ERROR


def cooling_down() -> float:
    """Seconds until model calls are worth trying again. 0 when ready."""
    return max(0.0, _RETRY_AFTER - time.monotonic())


def available() -> tuple[bool, Optional[str]]:
    """(usable, reason_if_not). Cheap enough to call on every render."""
    if config.NO_MODEL:
        return False, "disabled_by_TRUSTLENS_NO_MODEL"
    if not config.has_api_key():
        return False, "no_api_key"
    try:
        import google.generativeai  # noqa: F401
    except Exception:
        return False, "google_generativeai_not_installed"
    wait = cooling_down()
    if wait > 0:
        return False, f"{_BACKEND_ERROR} (retrying in {wait:.0f}s)"
    return True, None


def _model(model_name: str):
    import google.generativeai as genai
    genai.configure(api_key=config.api_key())
    return genai.GenerativeModel(model_name)


def extract_json(text: str) -> Optional[dict]:
    """Models wrap JSON in prose or code fences often enough to handle it."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def generate(prompt: str, *, model_name: Optional[str] = None,
             image=None, temperature: float = 0.2,
             max_tokens: int = 900) -> Optional[str]:
    """
    Single call, with fallback. Returns the text, or None if every model failed.

    On a rate limit we try the next model in config.MODEL_FALLBACKS rather than
    giving up, because free-tier quota is per model: one exhausted alias should
    not take the whole app down in the middle of a demo.
    """
    global _BACKEND_ERROR, _RETRY_AFTER
    ok, _ = available()
    if not ok:
        return None

    primary = model_name or config.TEXT_MODEL
    chain = [primary] + [m for m in config.MODEL_FALLBACKS if m != primary]
    parts: List[Any] = [prompt] + ([image] if image is not None else [])
    last_exc: Optional[Exception] = None

    for candidate in chain:
        try:
            resp = _model(candidate).generate_content(
                parts,
                generation_config={"temperature": temperature,
                                   "max_output_tokens": max_tokens},
            )
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                _BACKEND_ERROR = None
                _RETRY_AFTER = 0.0
                return text
        except Exception as exc:
            last_exc = exc
            kind = _classify_error(exc)
            if kind == "auth":
                break                      # a bad key fails on every model
            continue                       # rate limit or transient: try the next

    if last_exc is not None:
        # Keep the real reason. An earlier version stored only the exception
        # class name, so a retired model name surfaced as a bare "None" with no
        # way to tell it from a network failure or a bad key. The message is
        # from the API about the request, never about user content.
        detail = " ".join(str(last_exc).split())[:180]
        _BACKEND_ERROR = f"{type(last_exc).__name__}: {detail}" if detail else type(last_exc).__name__
        _RETRY_AFTER = time.monotonic() + _COOLDOWNS[_classify_error(last_exc)]
    return None


def generate_json(prompt: str, **kwargs) -> Optional[dict]:
    raw = generate(prompt, **kwargs)
    return extract_json(raw) if raw else None


def sample_json(prompt: str, n: int = None, *, temperature: float = 1.0,
                **kwargs) -> List[dict]:
    """
    Ask the same question n times at high temperature.

    High temperature is deliberate: we WANT the samples to diverge when the
    model is unsure, because that divergence is the measurement. The caller
    decides what counts as agreement.
    """
    n = config.VLM_SAMPLES if n is None else n
    out: List[dict] = []
    for _ in range(n):
        parsed = generate_json(prompt, temperature=temperature, **kwargs)
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def agreement(samples: List[dict], key: str, field: str) -> Dict[str, dict]:
    """
    Group repeated observations across samples by `field`.

    Returns {value: {hits, ratio, first}} so a caller can keep what a
    majority of samples agreed on and discard the rest as guesswork.
    """
    buckets: Dict[str, List[dict]] = {}
    for sample in samples:
        seen = set()
        for item in (sample.get(key) or []):
            if not isinstance(item, dict):
                continue
            val = str(item.get(field, "")).strip()
            if not val or val in seen:
                continue
            seen.add(val)
            buckets.setdefault(val, []).append(item)

    total = max(1, len(samples))
    return {
        val: {"hits": len(items), "ratio": round(len(items) / total, 2), "first": items[0]}
        for val, items in buckets.items()
    }
