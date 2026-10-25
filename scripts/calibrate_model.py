"""
Calibrate CLIP baseline image classifier for TrustLens using Platt scaling.

1. Loads cached CLIP ViT-Large/14 embeddings and labels from results/embeddings.npz.
2. Wraps the existing logistic regression model (models/image_clf.joblib) in
   sklearn's CalibratedClassifierCV with method="sigmoid" (Platt scaling) using 5-fold CV.
3. Computes out-of-fold probability predictions via 5-fold stratified cross-validation.
4. Generates a reliability diagram table partitioned into 10% probability bins (0-10% to 90-100%)
   displaying prediction counts, empirical fraction of actual AI images, and calibration gaps.
5. Fits the final calibrated classifier on all training embeddings and saves the model
   to models/image_clf_calibrated.joblib without overwriting models/image_clf.joblib.
"""
import os
import sys
import csv
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def compute_reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Bucket predictions into equal-width probability bins across [0, 1]
    and compute count, actual AI fraction, mean predicted probability,
    and calibration error for each bin.
    """
    bins_info = []
    total_samples = len(y_true)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        low = i / float(n_bins)
        high = (i + 1) / float(n_bins)
        label = f"{int(low * 100):2d}% - {int(high * 100):3d}%"

        # Last bin [0.90, 1.00] is closed on both ends
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))
        if count > 0:
            actual_ai_count = int(np.sum(y_true[mask]))
            actual_ai_frac = actual_ai_count / count
            mean_prob = float(np.mean(y_prob[mask]))
            cal_gap = mean_prob - actual_ai_frac
            abs_gap = abs(cal_gap)
            ece += (count / total_samples) * abs_gap
            mce = max(mce, abs_gap)
        else:
            actual_ai_count = 0
            actual_ai_frac = 0.0
            mean_prob = (low + high) / 2.0
            cal_gap = 0.0
            abs_gap = 0.0

        bins_info.append({
            "bin_idx": i + 1,
            "range": label,
            "low": low,
            "high": high,
            "count": count,
            "actual_ai": actual_ai_count,
            "actual_ai_frac": actual_ai_frac,
            "mean_prob": mean_prob,
            "gap": cal_gap,
            "abs_gap": abs_gap,
        })

    summary_metrics = {
        "ece": ece,
        "mce": mce,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "acc_05": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        "f1_05": float(f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0)),
    }
    return bins_info, summary_metrics


def print_reliability_table(bins_info: List[Dict], summary: Dict[str, float], title: str = "Out-of-Fold Reliability Table (10% Probability Bins)"):
    print("=" * 82)
    print(title)
    print("=" * 82)
    print(f"{'Bin Range':<12} {'Count':<7} {'Actual AI / Total':<18} {'Actual AI (%)':<15} {'Mean Pred (%)':<15} {'Gap (%)':<10}")
    print("-" * 82)
    total_count = sum(b["count"] for b in bins_info)
    total_ai = sum(b["actual_ai"] for b in bins_info)
    for b in bins_info:
        rng = b["range"]
        cnt = b["count"]
        if cnt > 0:
            ratio_str = f"{b['actual_ai']} / {cnt}"
            act_pct = f"{b['actual_ai_frac'] * 100:.2f}%"
            pred_pct = f"{b['mean_prob'] * 100:.2f}%"
            gap_pct = f"{b['gap'] * 100:+.2f}%"
        else:
            ratio_str = "0 / 0"
            act_pct = "0.00% (empty)"
            pred_pct = "-"
            gap_pct = "-"
        print(f"{rng:<12} {cnt:<7} {ratio_str:<18} {act_pct:<15} {pred_pct:<15} {gap_pct:<10}")

    print("-" * 82)
    overall_act_pct = (total_ai / total_count * 100) if total_count > 0 else 0.0
    tot_ratio_str = f"{total_ai} / {total_count}"
    tot_act_str = f"{overall_act_pct:.2f}%"
    ece_display = f"ECE: {summary['ece'] * 100:.2f}%"
    print(f"{'Total':<12} {total_count:<7} {tot_ratio_str:<18} {tot_act_str:<15} {'-':<15} {ece_display:<10}")
    print("=" * 82)


def calibrate_model(
    embeddings_path: str = "results/embeddings.npz",
    base_model_path: str = "models/image_clf.joblib",
    output_model_path: str = "models/image_clf_calibrated.joblib",
    output_oof_csv: Optional[str] = "results/oof_predictions_calibrated.csv",
    method: str = "sigmoid",
    n_splits: int = 5,
    random_seed: int = 42,
):
    # Resolve absolute paths
    embeddings_path = embeddings_path if os.path.isabs(embeddings_path) else os.path.join(REPO_ROOT, embeddings_path)
    base_model_path = base_model_path if os.path.isabs(base_model_path) else os.path.join(REPO_ROOT, base_model_path)
    output_model_path = output_model_path if os.path.isabs(output_model_path) else os.path.join(REPO_ROOT, output_model_path)
    if output_oof_csv and not os.path.isabs(output_oof_csv):
        output_oof_csv = os.path.join(REPO_ROOT, output_oof_csv)

    print("=" * 82)
    print("TrustLens Model Calibration & Reliability Analysis (Platt Scaling)")
    print("=" * 82)
    print(f"Embeddings cache:   {os.path.relpath(embeddings_path, REPO_ROOT).replace('\\', '/')}")
    print(f"Base model input:   {os.path.relpath(base_model_path, REPO_ROOT).replace('\\', '/')}")
    print(f"Calibrated output:  {os.path.relpath(output_model_path, REPO_ROOT).replace('\\', '/')}")
    print(f"Calibration method: {method.capitalize()} (Platt scaling)")
    print(f"Cross-validation:   {n_splits}-fold Stratified CV (seed={random_seed})\n")

    # Guard: do not overwrite base model
    if os.path.abspath(base_model_path) == os.path.abspath(output_model_path):
        raise ValueError(
            f"Output model path must be different from base model path. Overwrite prevented: {base_model_path}"
        )

    # 1. Load cached embeddings and labels
    if not os.path.isfile(embeddings_path):
        raise FileNotFoundError(
            f"Embeddings cache not found at {embeddings_path}. Run scripts/train_clip_baseline.py first."
        )

    print(f"Loading cached embeddings from {os.path.relpath(embeddings_path, REPO_ROOT).replace('\\', '/')}...")
    cached = np.load(embeddings_path, allow_pickle=True)
    X = cached["embeddings"]
    y = cached["labels"].astype(int)
    file_paths = [str(p) for p in cached["paths"]] if "paths" in cached else None

    n_real = int(np.sum(y == 0))
    n_ai = int(np.sum(y == 1))
    print(f"Loaded {len(y)} samples: {n_real} Real (label 0), {n_ai} AI (label 1); embedding dimension = {X.shape[1]}.")

    # 2. Load existing base model
    if not os.path.isfile(base_model_path):
        raise FileNotFoundError(
            f"Base model not found at {base_model_path}. Train the baseline model first."
        )

    print(f"Loading base classifier from {os.path.relpath(base_model_path, REPO_ROOT).replace('\\', '/')}...")
    base_clf = joblib.load(base_model_path)
    print(f"Base classifier: {type(base_clf).__name__}({base_clf})\n")

    # 3. Fit calibration layer with 5-fold CV to generate out-of-fold predictions
    print(f"Computing out-of-fold calibrated predictions using {n_splits}-fold Stratified CV...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    cal_cv = CalibratedClassifierCV(estimator=base_clf, method=method, cv=cv)
    oof_probs = cross_val_predict(cal_cv, X, y, cv=cv, method="predict_proba")[:, 1]
    print("Completed out-of-fold cross-validation.\n")

    # 4. Reliability table
    bins_info, summary = compute_reliability_table(y, oof_probs, n_bins=10)
    print_reliability_table(bins_info, summary, title="Out-of-Fold Reliability Table (10% Probability Bins)")

    print("\nCalibration & Performance Summary:")
    print(f"  Expected Calibration Error (ECE): {summary['ece']:.4f} ({summary['ece'] * 100:.2f}%)")
    print(f"  Maximum Calibration Error (MCE):  {summary['mce']:.4f} ({summary['mce'] * 100:.2f}%)")
    print(f"  Brier Score Loss:                 {summary['brier_score']:.4f}")
    print(f"  Out-of-Fold ROC AUC:              {summary['auc']:.4f}")
    print(f"  Out-of-Fold Accuracy @ 0.5:       {summary['acc_05']:.4f} ({summary['acc_05'] * 100:.2f}%)")
    print(f"  Out-of-Fold F1 Score:             {summary['f1_05']:.4f}")

    # Optional: Compare with uncalibrated OOF predictions if results/oof_predictions.csv exists
    oof_orig_path = os.path.join(REPO_ROOT, "results", "oof_predictions.csv")
    if os.path.isfile(oof_orig_path):
        try:
            with open(oof_orig_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                uncal_probs = np.array([float(r["probability"]) for r in reader])
            if len(uncal_probs) == len(y):
                _, uncal_summary = compute_reliability_table(y, uncal_probs, n_bins=10)
                print("\nComparison vs Uncalibrated Model:")
                print(f"  Uncalibrated ECE:         {uncal_summary['ece'] * 100:.2f}%  ->  Calibrated ECE: {summary['ece'] * 100:.2f}%")
                print(f"  Uncalibrated Brier Score: {uncal_summary['brier_score']:.4f}  ->  Calibrated Brier Score: {summary['brier_score']:.4f}")
        except Exception:
            pass

    # 5. Fit final calibrated model on all training data using 5-fold internal CV
    print(f"\nFitting final calibrated model on all {len(y)} samples ({n_splits}-fold internal CV)...")
    final_calibrated = CalibratedClassifierCV(
        estimator=base_clf,
        method=method,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed),
    )
    final_calibrated.fit(X, y)

    # 6. Save calibrated model to models/image_clf_calibrated.joblib
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(final_calibrated, output_model_path)
    print(f"Saved calibrated model to: {os.path.relpath(output_model_path, REPO_ROOT).replace('\\', '/')}")
    print(f"Original model preserved at: {os.path.relpath(base_model_path, REPO_ROOT).replace('\\', '/')}")

    # 7. Save out-of-fold predictions to CSV
    if output_oof_csv and file_paths:
        os.makedirs(os.path.dirname(output_oof_csv), exist_ok=True)
        with open(output_oof_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file path", "label", "probability"])
            for p, label, prob in zip(file_paths, y, oof_probs):
                writer.writerow([p, label, round(float(prob), 5)])
        print(f"Saved calibrated out-of-fold predictions to: {os.path.relpath(output_oof_csv, REPO_ROOT).replace('\\', '/')}")

    print("=" * 82)


def main():
    parser = argparse.ArgumentParser(description="TrustLens CLIP Baseline Model Calibration (Platt Scaling)")
    parser.add_argument(
        "--embeddings",
        default="results/embeddings.npz",
        help="Path to cached embeddings .npz file (default: results/embeddings.npz)",
    )
    parser.add_argument(
        "--base-model",
        default="models/image_clf.joblib",
        help="Path to trained base classifier (default: models/image_clf.joblib)",
    )
    parser.add_argument(
        "--output-model",
        default="models/image_clf_calibrated.joblib",
        help="Path to save calibrated model (default: models/image_clf_calibrated.joblib)",
    )
    parser.add_argument(
        "--output-oof",
        default="results/oof_predictions_calibrated.csv",
        help="Path to save out-of-fold calibrated predictions CSV (default: results/oof_predictions_calibrated.csv)",
    )
    parser.add_argument(
        "--method",
        default="sigmoid",
        choices=["sigmoid", "isotonic"],
        help="Calibration method: 'sigmoid' (Platt scaling) or 'isotonic' (default: sigmoid)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds for calibration (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for cross-validation splits (default: 42)",
    )
    args = parser.parse_args()

    calibrate_model(
        embeddings_path=args.embeddings,
        base_model_path=args.base_model,
        output_model_path=args.output_model,
        output_oof_csv=args.output_oof,
        method=args.method,
        n_splits=args.cv,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
