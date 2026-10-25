"""
Train CLIP (ViT-Large/14) image baseline classifier for TrustLens.

Data: data_processed/real/ (label 0) and data_processed/ai/ (label 1).
Features: L2-normalized image embeddings from openai/clip-vit-large-patch14.
Evaluation: 5-fold stratified cross-validation with nested inner tuning of C.
"""
import os
import sys
import csv
import warnings
from typing import List, Tuple

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from transformers import CLIPModel, CLIPProcessor

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_NAME = "openai/clip-vit-large-patch14"
RANDOM_SEED = 42
BATCH_SIZE = 16


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


def extract_or_load_embeddings(
    image_entries: List[Tuple[str, int]], cache_path: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load cached embeddings if available, else extract using openai/clip-vit-large-patch14
    and save to cache_path.
    """
    rel_paths = [os.path.relpath(p, REPO_ROOT).replace("\\", "/") for p, _ in image_entries]
    labels = np.array([l for _, l in image_entries], dtype=int)

    if os.path.isfile(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            cached_paths = [str(p) for p in cached["paths"]]
            if cached_paths == rel_paths and np.array_equal(cached["labels"], labels):
                print(f"Loaded cached embeddings from {os.path.relpath(cache_path, REPO_ROOT).replace('\\', '/')}")
                return cached["embeddings"], cached["labels"], rel_paths
            else:
                print("Cache mismatch detected. Re-extracting embeddings...")
        except Exception as exc:
            print(f"Failed to read cache ({exc}). Re-extracting embeddings...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading CLIP model ({MODEL_NAME}) on {device.upper()}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"Extracting L2-normalized image embeddings for {len(image_entries)} images (batch size {BATCH_SIZE})...")

    embeddings_list = []
    total = len(image_entries)

    for i in range(0, total, BATCH_SIZE):
        batch_entries = image_entries[i : i + BATCH_SIZE]
        batch_images = []
        for p, _ in batch_entries:
            with Image.open(p) as img:
                batch_images.append(img.convert("RGB"))

        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
            if hasattr(feat, "pooler_output"):
                feat = feat.pooler_output
            elif isinstance(feat, (tuple, list)):
                feat = feat[0]
            # L2 normalization
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embeddings_list.append(feat.cpu().numpy())

        done = min(i + BATCH_SIZE, total)
        print(f"  Extracted {done}/{total} embeddings...")

    embeddings = np.concatenate(embeddings_list, axis=0)

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings, labels=labels, paths=np.array(rel_paths))
    print(f"Saved embeddings cache to {os.path.relpath(cache_path, REPO_ROOT).replace('\\', '/')}\n")

    return embeddings, labels, rel_paths


def tpr_at_target_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05) -> float:
    """Compute true positive rate at specified false positive rate (default 5%)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(target_fpr, fpr, tpr))


def evaluate_cv(
    X: np.ndarray, y: np.ndarray, C_candidates: List[float], random_seed: int = 42
) -> Tuple[List[dict], np.ndarray]:
    """Run 5-fold stratified cross-validation with nested inner tuning of C."""
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    oof_probs = np.zeros(len(y), dtype=float)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Inner cross-validation on training fold only (no leakage)
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_seed)
        grid = GridSearchCV(
            LogisticRegression(solver="liblinear", random_state=random_seed),
            param_grid={"C": C_candidates},
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        grid.fit(X_train, y_train)
        best_clf = grid.best_estimator_

        # Predict out-of-fold probabilities
        val_probs = best_clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_probs

        val_preds_binary = (val_probs >= 0.5).astype(int)
        auc = roc_auc_score(y_val, val_probs)
        acc = accuracy_score(y_val, val_preds_binary)
        f1 = f1_score(y_val, val_preds_binary, zero_division=0)
        tpr_5fpr = tpr_at_target_fpr(y_val, val_probs, target_fpr=0.05)

        fold_metrics.append({
            "fold": fold,
            "best_C": grid.best_params_["C"],
            "auc": auc,
            "acc": acc,
            "f1": f1,
            "tpr@5%fpr": tpr_5fpr,
        })

    return fold_metrics, oof_probs


def main():
    processed_dir = os.path.join(REPO_ROOT, "data_processed")
    real_dir = os.path.join(processed_dir, "real")
    ai_dir = os.path.join(processed_dir, "ai")
    results_dir = os.path.join(REPO_ROOT, "results")
    models_dir = os.path.join(REPO_ROOT, "models")
    cache_path = os.path.join(results_dir, "embeddings.npz")
    oof_csv_path = os.path.join(results_dir, "oof_predictions.csv")
    model_save_path = os.path.join(models_dir, "image_clf.joblib")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 72)
    print("TrustLens CLIP (ViT-Large/14) Baseline Classifier")
    print("=" * 72)

    image_entries = find_images(real_dir, label=0) + find_images(ai_dir, label=1)
    if not image_entries:
        print("Error: No images found under data_processed/real/ or data_processed/ai/.")
        print("Run scripts/prepare_dataset.py first.")
        return

    n_real = sum(1 for _, l in image_entries if l == 0)
    n_ai = sum(1 for _, l in image_entries if l == 1)
    print(f"Dataset: {len(image_entries)} images ({n_real} real, {n_ai} AI)")

    # 1. Extract or load cached embeddings
    X, y, file_paths = extract_or_load_embeddings(image_entries, cache_path)

    # 2. 5-fold stratified cross-validation with inner tuning of C
    C_candidates = [0.01, 0.1, 1.0, 10.0]
    print("Running 5-fold stratified cross-validation with nested tuning of C in [0.01, 0.1, 1, 10]...")
    fold_metrics, oof_probs = evaluate_cv(X, y, C_candidates, random_seed=RANDOM_SEED)

    # 3. Report per fold and averaged
    print("\n" + "-" * 72)
    print(f"{'Fold':<6} {'Best C':<8} {'AUC':<10} {'Accuracy@0.5':<15} {'F1':<10} {'TPR @ 5% FPR':<12}")
    print("-" * 72)
    for m in fold_metrics:
        print(f"Fold {m['fold']:<2} {m['best_C']:<8} {m['auc']:<10.4f} {m['acc']:<15.4f} {m['f1']:<10.4f} {m['tpr@5%fpr']:<12.4f}")

    avg_auc = float(np.mean([m["auc"] for m in fold_metrics]))
    avg_acc = float(np.mean([m["acc"] for m in fold_metrics]))
    avg_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
    avg_tpr = float(np.mean([m["tpr@5%fpr"] for m in fold_metrics]))

    print("-" * 72)
    print(f"{'Mean':<6} {'-':<8} {avg_auc:<10.4f} {avg_acc:<15.4f} {avg_f1:<10.4f} {avg_tpr:<12.4f}")
    print("-" * 72)

    # 4. Save out-of-fold predicted probabilities
    with open(oof_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file path", "label", "probability"])
        for p, label, prob in zip(file_paths, y, oof_probs):
            writer.writerow([p, label, round(float(prob), 5)])
    print(f"\nSaved out-of-fold predictions to {os.path.relpath(oof_csv_path, REPO_ROOT).replace('\\', '/')}")

    # 5. Comparison table of AUC
    print("\n" + "=" * 72)
    print("AUC Performance Comparison Across Approaches")
    print("=" * 72)
    print(f"{'Approach':<42} {'AUC':<10} {'Notes':<20}")
    print("-" * 72)
    print(f"{'TrustLens Forensic Score (Baseline)':<42} {'0.538':<10} {'Forensics only, no model'}")
    print(f"{'Metadata-Only Logistic Regression (Normalized)':<42} {'0.608':<10} {'Shortcuts neutralized'}")
    print(f"{'CLIP ViT-Large/14 + Logistic Regression':<42} {f'{avg_auc:.4f}':<10} {'Content semantic representations'}")
    print("=" * 72)

    # 6. Sanity check: repeat CV with randomly shuffled labels
    print("\nSanity Check: Running 5-fold CV with randomly shuffled labels...")
    rng = np.random.RandomState(RANDOM_SEED)
    y_shuffled = rng.permutation(y)
    shuffled_metrics, _ = evaluate_cv(X, y_shuffled, C_candidates, random_seed=RANDOM_SEED)
    shuffled_auc = float(np.mean([m["auc"] for m in shuffled_metrics]))
    print(f"Shuffled Labels Average AUC: {shuffled_auc:.4f}")

    if abs(shuffled_auc - 0.5) > 0.15:
        print(f"  -> WARNING: Shuffled AUC ({shuffled_auc:.4f}) deviates significantly from 0.50!")
    else:
        print("  -> PASSED: Shuffled AUC is near 0.50 (chance level), confirming no data leakage.")

    # 7. Fit final model on all data and save to models/image_clf.joblib
    print("\nFitting final model on all data with inner CV for C selection...")
    inner_cv_all = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid_all = GridSearchCV(
        LogisticRegression(solver="liblinear", random_state=RANDOM_SEED),
        param_grid={"C": C_candidates},
        cv=inner_cv_all,
        scoring="roc_auc",
        n_jobs=1,
    )
    grid_all.fit(X, y)
    final_model = grid_all.best_estimator_
    print(f"Selected best C = {grid_all.best_params_['C']} for final model.")

    joblib.dump(final_model, model_save_path)
    print(f"Saved final trained classifier to {os.path.relpath(model_save_path, REPO_ROOT).replace('\\', '/')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
