"""
Baseline evaluation script for TrustLens.

Runs offline, forensic-only analysis on images in data/real/ and data/ai/,
saves per-image results to results/baseline.csv, and prints summary metrics.
"""
import os
import sys
import csv
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Run offline, forensic-only: no Gemini API calls
os.environ["TRUSTLENS_NO_MODEL"] = "1"

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from trustlens import detectors
from trustlens.schemas import Verdict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

VERDICT_TO_BAND = {
    Verdict.DECLARED_AI: "Declared AI",
    Verdict.LIKELY_AUTHENTIC: "Likely Authentic",
    Verdict.UNCERTAIN: "Uncertain",
    Verdict.LIKELY_MANIPULATED: "Likely Manipulated",
}

BANDS = ["Declared AI", "Likely Authentic", "Uncertain", "Likely Manipulated"]


def find_images(base_dir: str, true_label: int, is_ai: bool) -> List[Tuple[str, int, str]]:
    """
    Find all image files under base_dir.
    Returns list of (filepath, true_label, generator).
    """
    items = []
    if not os.path.isdir(base_dir):
        return items

    for root, _, files in os.walk(base_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, base_dir)
                parts = os.path.normpath(rel_path).split(os.sep)

                if is_ai:
                    # Subfolder name is generator
                    generator = parts[0] if len(parts) > 1 else "ai"
                else:
                    # Subfolder name if present, else "real"
                    generator = parts[0] if len(parts) > 1 else "real"

                items.append((full_path, true_label, generator))

    # Sort deterministically
    items.sort(key=lambda x: x[0])
    return items


def evaluate():
    real_dir = os.path.join(REPO_ROOT, "data", "real")
    ai_dir = os.path.join(REPO_ROOT, "data", "ai")
    results_dir = os.path.join(REPO_ROOT, "results")
    csv_path = os.path.join(results_dir, "baseline.csv")

    os.makedirs(results_dir, exist_ok=True)

    images = find_images(real_dir, true_label=0, is_ai=False) + \
             find_images(ai_dir, true_label=1, is_ai=True)

    print(f"Discovered {len(images)} images ({sum(1 for _, l, _ in images if l == 0)} real, "
          f"{sum(1 for _, l, _ in images if l == 1)} ai).")
    print("Running TrustLens image analysis (offline, forensic-only)...\n")

    # CSV headers
    fieldnames = [
        "file path",
        "true label",
        "generator",
        "final band",
        "final score",
        "ELA",
        "noise residual",
        "JPEG ghost",
        "provenance",
        "jpeg_compression",
        "exif",
        "faces_detected",
        "signals",
        "error",
    ]

    # Metrics trackers
    band_counts = {
        "real": {b: 0 for b in BANDS},
        "ai": {b: 0 for b in BANDS},
    }
    generator_likely_auth = defaultdict(int)
    generator_total = defaultdict(int)

    y_true: List[int] = []
    y_scores: List[float] = []
    total_errors = 0

    rows = []

    for idx, (filepath, label, generator) in enumerate(images, start=1):
        rel_display_path = os.path.relpath(filepath, REPO_ROOT).replace("\\", "/")
        generator_total[generator] += 1

        try:
            with open(filepath, "rb") as fh:
                payload = fh.read()

            # Analyze exactly as api.py does
            evidence = detectors.analyse(payload, filename=os.path.basename(filepath))

            band = VERDICT_TO_BAND.get(evidence.verdict, "Uncertain")
            score = round(float(evidence.score), 4)

            findings = evidence.findings or {}
            provenance = evidence.provenance or {}

            # Extract signal values
            ela_val = findings.get("ela", {}).get("max_z")
            noise_val = findings.get("noise", {}).get("max_z")
            ghost_val = findings.get("ghost", {}).get("max_deviation_steps")
            prov_val = provenance.get("method") or provenance.get("detail") or (
                "declared_ai" if provenance.get("declared_ai") else "none"
            )
            jpeg_val = findings.get("jpeg_quality")
            exif_val = findings.get("exif_tags")
            faces_val = findings.get("faces_detected", 0)

            signals_summary = "; ".join(f"{s.name}={s.value}" for s in evidence.signals)

            row = {
                "file path": rel_display_path,
                "true label": label,
                "generator": generator,
                "final band": band,
                "final score": score,
                "ELA": ela_val if ela_val is not None else "",
                "noise residual": noise_val if noise_val is not None else "",
                "JPEG ghost": ghost_val if ghost_val is not None else "",
                "provenance": prov_val,
                "jpeg_compression": jpeg_val if jpeg_val is not None else "",
                "exif": exif_val if exif_val is not None else "",
                "faces_detected": faces_val if faces_val is not None else "",
                "signals": signals_summary,
                "error": "",
            }

            band_counts["ai" if label == 1 else "real"][band] += 1
            if band == "Likely Authentic":
                generator_likely_auth[generator] += 1

            y_true.append(label)
            y_scores.append(score)

        except Exception as exc:
            total_errors += 1
            error_msg = f"{type(exc).__name__}: {exc}"
            band = "Uncertain"
            row = {
                "file path": rel_display_path,
                "true label": label,
                "generator": generator,
                "final band": band,
                "final score": "",
                "ELA": "",
                "noise residual": "",
                "JPEG ghost": "",
                "provenance": "",
                "jpeg_compression": "",
                "exif": "",
                "faces_detected": "",
                "signals": "",
                "error": error_msg,
            }
            band_counts["ai" if label == 1 else "real"][band] += 1

        rows.append(row)

        if idx % 25 == 0 or idx == len(images):
            print(f"  Processed {idx}/{len(images)} images...")

    # Write to results/baseline.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {os.path.relpath(csv_path, REPO_ROOT).replace('\\', '/')}\n")

    # -----------------------------------------------------------------------
    # Summary reporting
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("TrustLens Baseline Evaluation Summary")
    print("=" * 60)

    print("\nCount of images in each band:")
    print("  Real images (true label = 0):")
    for band in BANDS:
        count = band_counts["real"][band]
        print(f"    - {band:20s}: {count}")

    print("\n  AI images (true label = 1):")
    for band in BANDS:
        count = band_counts["ai"][band]
        print(f"    - {band:20s}: {count}")

    print("\nLikely Authentic count by generator folder:")
    for gen in sorted(generator_total.keys()):
        auth_count = generator_likely_auth[gen]
        total = generator_total[gen]
        pct = (auth_count / total * 100) if total > 0 else 0.0
        print(f"  - {gen}: {auth_count}/{total} called Likely Authentic ({pct:.1f}%)")

    print("\nOverall AUC (real vs ai):")
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(y_true)) > 1 and len(y_scores) == len(y_true):
            auc = roc_auc_score(y_true, y_scores)
            print(f"  AUC: {auc:.4f}")
        else:
            print("  AUC could not be calculated (insufficient class variety).")
    except ImportError:
        print("  scikit-learn is not installed (cannot compute AUC).")

    if total_errors > 0:
        print(f"\nEncountered {total_errors} errors during analysis (recorded in CSV).")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
