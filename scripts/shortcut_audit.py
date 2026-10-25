"""
Dataset shortcut audit script for TrustLens.

Audits dataset bias and shortcut features between real and AI images.
Measures file metadata without looking at pixel content, trains a 5-fold
cross-validated logistic regression on metadata alone, and compares its AUC
to TrustLens's baseline forensic score.
"""
import os
import sys
import csv
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from trustlens.detectors.forensics_image import jpeg_quality_from_qtables

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ===========================================================================
# AUC calculation & simple Logistic Regression (NumPy fallback / scikit-learn)
# ===========================================================================
def compute_roc_auc(y_true: List[int], y_score: List[float]) -> float:
    """
    Compute ROC AUC via Mann-Whitney U rank statistics (handles ties via mid-ranks).
    Matches sklearn.metrics.roc_auc_score exactly.
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        pass

    y_arr = np.asarray(y_true, dtype=bool)
    score_arr = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_arr))
    n_neg = len(y_arr) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(score_arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score_arr) + 1)

    unique_scores, counts = np.unique(score_arr, return_counts=True)
    for score, count in zip(unique_scores, counts):
        if count > 1:
            idx = np.where(score_arr == score)[0]
            ranks[idx] = ranks[idx].mean()

    u_stat = float(np.sum(ranks[y_arr])) - (n_pos * (n_pos + 1)) / 2.0
    return u_stat / (n_pos * n_neg)


class SimpleLogisticRegression:
    """Standard L2-regularized logistic regression trained via gradient descent."""
    def __init__(self, lr: float = 0.05, n_iter: int = 1000, reg: float = 0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.reg = reg
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.n_iter):
            linear = np.clip(np.dot(X, self.weights) + self.bias, -25.0, 25.0)
            preds = 1.0 / (1.0 + np.exp(-linear))

            dw = (1.0 / n_samples) * np.dot(X.T, (preds - y)) + self.reg * self.weights
            db = (1.0 / n_samples) * np.sum(preds - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear = np.clip(np.dot(X, self.weights) + self.bias, -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-linear))


# ===========================================================================
# 1. Dataset collection & metadata extraction
# ===========================================================================
def find_images(base_dir: str, label: int) -> List[Tuple[str, int]]:
    items = []
    if not os.path.isdir(base_dir):
        return items

    for root, _, files in os.walk(base_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                full_path = os.path.join(root, f)
                items.append((full_path, label))

    items.sort(key=lambda x: x[0])
    return items


def audit(data_dir: str = "data"):
    real_dir = os.path.join(REPO_ROOT, data_dir, "real")
    ai_dir = os.path.join(REPO_ROOT, data_dir, "ai")
    results_dir = os.path.join(REPO_ROOT, "results")
    csv_path = os.path.join(results_dir, "shortcut_audit.csv")
    baseline_csv_path = os.path.join(results_dir, "baseline.csv")

    os.makedirs(results_dir, exist_ok=True)

    image_entries = find_images(real_dir, label=0) + find_images(ai_dir, label=1)
    print(f"Auditing metadata for {len(image_entries)} images from '{data_dir}' "
          f"({sum(1 for _, l in image_entries if l == 0)} real, "
          f"{sum(1 for _, l in image_entries if l == 1)} ai)...\n")

    if not image_entries:
        print(f"Error: No images found under {data_dir}/real or {data_dir}/ai.")
        return

    records = []

    for filepath, label in image_entries:
        rel_path = os.path.relpath(filepath, REPO_ROOT).replace("\\", "/")
        file_ext = os.path.splitext(filepath)[1].lower()
        file_size_bytes = os.path.getsize(filepath)
        file_size_kb = round(file_size_bytes / 1024.0, 2)

        # Open image with Pillow only, do not modify
        with Image.open(filepath) as img:
            actual_format = str(img.format or "").upper()
            width, height = img.size
            total_pixels = width * height
            megapixels = round(total_pixels / 1_000_000.0, 4)
            bytes_per_pixel = round(file_size_bytes / float(total_pixels), 4)
            aspect_ratio = round(width / float(height), 4) if height > 0 else 1.0

            # EXIF check
            try:
                exif = img.getexif()
                has_exif = bool(exif and len(exif) > 0)
            except Exception:
                has_exif = False

            # JPEG quality estimation from quantization tables
            if actual_format == "JPEG":
                q_est = jpeg_quality_from_qtables(img)
            else:
                q_est = None

        record = {
            "file path": rel_path,
            "label": label,
            "file extension": file_ext,
            "actual image format": actual_format,
            "width": width,
            "height": height,
            "aspect ratio": aspect_ratio,
            "megapixels": megapixels,
            "file size in KB": file_size_kb,
            "bytes per pixel": bytes_per_pixel,
            "whether EXIF is present": has_exif,
            "estimated jpeg quality": q_est if q_est is not None else "",
        }
        records.append(record)

    # -----------------------------------------------------------------------
    # 2. Save everything to results/shortcut_audit.csv
    # -----------------------------------------------------------------------
    fieldnames = [
        "file path",
        "label",
        "file extension",
        "actual image format",
        "width",
        "height",
        "aspect ratio",
        "megapixels",
        "file size in KB",
        "bytes per pixel",
        "whether EXIF is present",
        "estimated jpeg quality",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved audit records to {os.path.relpath(csv_path, REPO_ROOT).replace('\\', '/')}\n")

    # -----------------------------------------------------------------------
    # 3. Print summary comparing real vs ai
    # -----------------------------------------------------------------------
    real_records = [r for r in records if r["label"] == 0]
    ai_records = [r for r in records if r["label"] == 1]

    def stats(values: List[float]) -> Tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        arr = np.array(values, dtype=float)
        return float(np.median(arr)), float(np.min(arr)), float(np.max(arr))

    print("=" * 68)
    print(f"TrustLens Dataset Shortcut Audit Summary (Data: {data_dir})")
    print("=" * 68)

    # Format counts
    real_fmts = {}
    for r in real_records:
        real_fmts[r["actual image format"]] = real_fmts.get(r["actual image format"], 0) + 1
    ai_fmts = {}
    for r in ai_records:
        ai_fmts[r["actual image format"]] = ai_fmts.get(r["actual image format"], 0) + 1

    print("\nFormat Counts:")
    print("  Real: " + (", ".join(f"{k}: {v}" for k, v in sorted(real_fmts.items())) if real_fmts else "(none)"))
    print("  AI:   " + (", ".join(f"{k}: {v}" for k, v in sorted(ai_fmts.items())) if ai_fmts else "(none)"))

    # EXIF presence
    real_exif_pct = (sum(1 for r in real_records if r["whether EXIF is present"]) / len(real_records) * 100.0) if real_records else 0.0
    ai_exif_pct = (sum(1 for r in ai_records if r["whether EXIF is present"]) / len(ai_records) * 100.0) if ai_records else 0.0
    print(f"\nEXIF Metadata Presence:")
    print(f"  Real: {real_exif_pct:.1f}% ({sum(1 for r in real_records if r['whether EXIF is present'])} / {len(real_records)})")
    print(f"  AI:   {ai_exif_pct:.1f}% ({sum(1 for r in ai_records if r['whether EXIF is present'])} / {len(ai_records)})")

    # Dimensional and size statistics (median [min - max])
    metrics = [
        ("Width (px)", [r["width"] for r in real_records], [r["width"] for r in ai_records]),
        ("Height (px)", [r["height"] for r in real_records], [r["height"] for r in ai_records]),
        ("Aspect Ratio", [r["aspect ratio"] for r in real_records], [r["aspect ratio"] for r in ai_records]),
        ("File Size (KB)", [r["file size in KB"] for r in real_records], [r["file size in KB"] for r in ai_records]),
        ("Bytes per Pixel", [r["bytes per pixel"] for r in real_records], [r["bytes per pixel"] for r in ai_records]),
    ]

    print("\nFile & Dimension Distributions (Median [Min - Max]):")
    for name, r_vals, a_vals in metrics:
        r_med, r_min, r_max = stats(r_vals)
        a_med, a_min, a_max = stats(a_vals)
        print(f"  {name}:")
        print(f"    Real: {r_med:.2f} [{r_min:.2f} - {r_max:.2f}]")
        print(f"    AI:   {a_med:.2f} [{a_min:.2f} - {a_max:.2f}]")

    # JPEG Quality comparison (JPEGs only)
    real_qs = [r["estimated jpeg quality"] for r in real_records if r["estimated jpeg quality"] != ""]
    ai_qs = [r["estimated jpeg quality"] for r in ai_records if r["estimated jpeg quality"] != ""]
    if real_qs or ai_qs:
        r_med, r_min, r_max = stats(real_qs)
        a_med, a_min, a_max = stats(ai_qs)
        print(f"  Estimated JPEG Quality (JPEGs only):")
        print(f"    Real: {r_med:.1f} [{r_min:.1f} - {r_max:.1f}]")
        print(f"    AI:   {a_med:.1f} [{a_min:.1f} - {a_max:.1f}]")

    # -----------------------------------------------------------------------
    # 4. Train simple logistic regression on metadata columns only (5-fold CV)
    # -----------------------------------------------------------------------
    # Construct feature matrix X from metadata columns only:
    #   - width, height, aspect ratio, megapixels, file size in KB, bytes per pixel
    #   - whether EXIF is present
    #   - format indicators (is_jpeg, is_png, is_webp)
    #   - estimated jpeg quality and presence indicator
    feature_rows = []
    labels = []

    for r in records:
        fmt = r["actual image format"]
        q_val = float(r["estimated jpeg quality"]) if r["estimated jpeg quality"] != "" else 0.0
        has_q = 1.0 if r["estimated jpeg quality"] != "" else 0.0

        feat = [
            float(r["width"]),
            float(r["height"]),
            float(r["aspect ratio"]),
            float(r["megapixels"]),
            float(r["file size in KB"]),
            float(r["bytes per pixel"]),
            1.0 if r["whether EXIF is present"] else 0.0,
            1.0 if fmt == "JPEG" else 0.0,
            1.0 if fmt == "PNG" else 0.0,
            1.0 if fmt == "WEBP" else 0.0,
            q_val,
            has_q,
        ]
        feature_rows.append(feat)
        labels.append(int(r["label"]))

    X = np.array(feature_rows, dtype=float)
    y = np.array(labels, dtype=int)

    # 5-fold cross-validation
    np.random.seed(42)
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    pos_folds = np.array_split(pos_indices, 5)
    neg_folds = np.array_split(neg_indices, 5)

    cv_aucs = []

    try:
        from sklearn.linear_model import LogisticRegression
        use_sklearn = True
    except ImportError:
        use_sklearn = False

    for fold in range(5):
        val_idx = np.concatenate([pos_folds[fold], neg_folds[fold]])
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Standardize features using train fold statistics
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0

        X_train_std = (X_train - mean) / std
        X_val_std = (X_val - mean) / std

        if use_sklearn:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                clf = LogisticRegression(C=1.0, solver="liblinear", random_state=42)
                clf.fit(X_train_std, y_train)
                probs = clf.predict_proba(X_val_std)[:, 1]
        else:
            clf = SimpleLogisticRegression(lr=0.05, n_iter=1000, reg=0.01)
            clf.fit(X_train_std, y_train)
            probs = clf.predict_proba(X_val_std)

        fold_auc = compute_roc_auc(y_val, probs)
        cv_aucs.append(fold_auc)

    avg_meta_auc = float(np.mean(cv_aucs))

    print("\n" + "-" * 68)
    print("Metadata-Only Logistic Regression (5-fold Cross-Validation):")
    print("-" * 68)
    print(f"  Fold AUCs:   {[round(a, 4) for a in cv_aucs]}")
    print(f"  Average AUC: {avg_meta_auc:.4f}")

    # =======================================================================
    # EXPLANATION OF SHORTCUT LEARNING:
    # An AUC well above 0.5 (e.g. 0.8+ or ~0.95) means the dataset has severe
    # shortcuts. This proves that a classifier can distinguish "real" from
    # "AI" images with high accuracy using ONLY superficial container
    # properties (dimensions, file size, bytes-per-pixel, compression quality,
    # and EXIF presence) WITHOUT EVER LOOKING AT PIXEL CONTENT or learning
    # actual generative artifacts. A model trained on such data will overfit
    # to these non-causal metadata artifacts and fail on real-world test sets.
    # Conversely, an AUC near 0.50 (e.g. ~0.50 - 0.60) demonstrates that the
    # normalization succeeded in removing metadata shortcuts.
    # =======================================================================
    if avg_meta_auc > 0.70:
        print(f"  -> WARNING: Average metadata AUC ({avg_meta_auc:.4f}) is well above 0.50!")
        print("     The dataset contains strong shortcuts: superficial file metadata alone")
        print("     distinguishes real from AI without examining any pixel content.")
    else:
        print(f"  -> SUCCESS: Average metadata AUC ({avg_meta_auc:.4f}) is close to 0.50 (chance level).")
        print("     Superficial metadata shortcuts have been neutralized.")

    # -----------------------------------------------------------------------
    # 5. Compute and print the AUC of the TrustLens baseline score
    # -----------------------------------------------------------------------
    print("\n" + "-" * 68)
    print("TrustLens Baseline Final Score Comparison (results/baseline.csv):")
    print("-" * 68)

    if os.path.isfile(baseline_csv_path):
        base_y_true = []
        base_y_score = []
        with open(baseline_csv_path, "r", encoding="utf-8") as bf:
            reader = csv.DictReader(bf)
            for row in reader:
                score_str = row.get("final score", "").strip()
                label_str = row.get("true label", "").strip()
                if score_str and label_str:
                    try:
                        base_y_score.append(float(score_str))
                        base_y_true.append(int(label_str))
                    except ValueError:
                        pass

        if len(base_y_true) > 0 and len(set(base_y_true)) > 1:
            baseline_auc = compute_roc_auc(base_y_true, base_y_score)
            print(f"  TrustLens Forensic Final Score AUC: {baseline_auc:.4f} (across {len(base_y_true)} images)")
        else:
            print("  Insufficient valid scores in results/baseline.csv to compute AUC.")
    else:
        print(f"  {baseline_csv_path} not found. Run scripts/baseline_eval.py first.")

    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset shortcut audit for TrustLens")
    parser.add_argument("--data-dir", default="data",
                        help="Root data directory containing real/ and ai/ (default: 'data')")
    args = parser.parse_args()
    audit(data_dir=args.data_dir)
