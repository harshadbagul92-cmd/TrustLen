# Truth Lens - Agent Rules & Architecture Guidelines

## Core Principles & Design Mandate

> [!IMPORTANT]
> **No Binary Verdicts**: Published deepfake detectors collapse on real-world data (Deepfake-Eval-2024 benchmark shows open-source AUC dropping from 0.87-1.00 down to 0.43-0.63 in-the-wild).
> Truth Lens **NEVER** returns a binary "real vs fake" verdict.

### Four Verdict Bands
1. **`declared_ai`**: Provenance metadata (C2PA, EXIF, header tags) explicitly certifies AI generation.
2. **`likely_authentic`**: Calibrated score < 0.35. Phrase strictly as **"no evidence of manipulation"**, NEVER as "this is real" or "authentic".
3. **`uncertain`**: Score 0.35–0.65, **OR** any score when an out-of-distribution (OOD) reliability flag is triggered.
4. **`likely_manipulated`**: Calibrated score > 0.65.

---

## Lane Boundaries & Parallel Workflows

- **Member A**: Owns `Text` + `Audio` detection lanes (`detectors/text.py`, `detectors/audio.py`).
- **Member B**: Owns `Image` + `Video` detection lanes (`detectors/image.py`, `detectors/video.py`).

**Rule**: Member A and Member B files must remain strictly decoupled. All lanes implement the common base interface `analyse(input, metadata) -> Evidence`.

---

## Schema Contract (`schemas.py`)

Every detection lane must produce an `Evidence` object with:
- `modality`: `"text" | "image" | "audio" | "video"`
- `verdict`: `"declared_ai" | "likely_authentic" | "uncertain" | "likely_manipulated"`
- `score`: Calibrated `float` (0.0 to 1.0)
- `confidence`: `Tuple[float, float]` (low, high interval, widened by reliability gate)
- `signals`: `List[Signal]` where each signal has `name`, `human`, `value`, `weight`, and optional `where` (line number, timestamp range, or spatial region)
- `reliability`: `Reliability` object (`ood_flags`, `band` float widening, `note`)
- `provenance`: `dict` of header/metadata findings
- `explanation`: `str` filled last by `explainer.py`

---

## Shared Explainer Constraints (`explainer.py`)

- `explain(evidence: Evidence) -> str`
- Prompt LLM with a single consistent template.
- **Strict Requirement**: The explainer may reference **ONLY** signals present in the `Evidence` object. It must **NEVER** invent ungrounded evidence or external assertions.
