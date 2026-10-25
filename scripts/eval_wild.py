"""
Evaluate CLIP baseline classifier on the wild test dataset for TrustLens.

Loads models/image_clf.joblib and openai/clip-vit-large-patch14, embeds
images in data_wild_processed/ using extract_or_load_embeddings from
scripts.train_clip_baseline, recovers original subfolders from mapping CSV,
saves predictions to results/wild_predictions.csv, and reports metrics.
"""
import os
import sys
import csv
import argparse
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.train_clip_baseline import IMAGE_EXTENSIONS, extract_or_load_embeddings


def extract_subfolder(path_str: str) -> str:
    """
    Extract subfolder category (e.g. real/stock, ai/mundane) from image path.
    """
    norm = os.path.normpath(path_str).replace("\\", "/")
    parts = norm.split("/")
    for token in ["real", "ai"]:
        if token in parts:
            idx = parts.index(token)
            subparts = parts[idx:-1]
            if subparts:
                return "/".join(subparts)
            return token

    parent = os.path.dirname(norm)
    return parent if parent else "unknown"


def load_mapping_dict(mapping_csv_path: str) -> Dict[str, dict]:
    """
    Read mapping CSV into dictionary keyed by normalized paths and filenames.
    """
    mapping = {}
    if not os.path.isfile(mapping_csv_path):
        return mapping

    with open(mapping_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_path = row.get("new path", "")
            orig_path = row.get("original path", "")
            new_name = row.get("new name", "")
            label_val = int(row.get("label", 0))

            subfolder = extract_subfolder(orig_path)
            if subfolder == "unknown" and new_path:
                subfolder = extract_subfolder(new_path)

            entry = {
                "new_path": new_path,
                "orig_path": orig_path,
                "label": label_val,
                "subfolder": subfolder,
            }

            norm_new = os.path.normpath(new_path).replace("\\", "/")
            mapping[norm_new] = entry
            if new_name:
                mapping[new_name] = entry

    return mapping


def discover_wild_images(
    data_dir: str, mapping: Dict[str, dict]
) -> Tuple[List[Tuple[str, int]], List[str], List[dict]]:
    """
    Find all images in data_dir, matching with mapping CSV where available.
    Returns:
      image_entries: List of (full_path, label)
      subfolders: List of recovered subfolders (e.g. real/stock, ai/mundane)
      metadata_rows: List of metadata dicts
    """
    found_paths = []
    if os.path.isdir(data_dir):
        for root, _, files in os.walk(data_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    found_paths.append(os.path.join(root, f))

    found_paths.sort()

    image_entries = []
    subfolders = []
    metadata_rows = []

    for full_path in found_paths:
        rel_to_repo = os.path.relpath(full_path, REPO_ROOT).replace("\\", "/")
        fname = os.path.basename(full_path)

        # Lookup in mapping CSV
        map_entry = mapping.get(rel_to_repo) or mapping.get(fname)
        if map_entry:
            label = map_entry["label"]
            subfolder = map_entry["subfolder"]
            orig_path = map_entry["orig_path"]
        else:
            # Fallback if mapping entry is missing
            norm_rel = rel_to_repo.lower()
            label = 0 if "/real/" in norm_rel or norm_rel.startswith("real/") else 1
            subfolder = extract_subfolder(rel_to_repo)
            orig_path = rel_to_repo

        image_entries.append((full_path, label))
        subfolders.append(subfolder)
        metadata_rows.append({
            "full_path": full_path,
            "rel_path": rel_to_repo,
            "subfolder": subfolder,
            "orig_path": orig_path,
            "label": label,
        })

    return image_entries, subfolders, metadata_rows


def evaluate_wild(
    data_dir: str = "data_wild_processed",
    mapping_path: Optional[str] = None,
    model_path: Optional[str] = "models/image_clf.joblib",
    cache_path: Optional[str] = None,
    output_csv: Optional[str] = None,
):
    data_dir_abs = data_dir if os.path.isabs(data_dir) else os.path.join(REPO_ROOT, data_dir)
    results_dir = os.path.join(REPO_ROOT, "results")
    models_dir = os.path.join(REPO_ROOT, "models")
    os.makedirs(results_dir, exist_ok=True)

    if not model_path:
        model_path = os.path.join(models_dir, "image_clf.joblib")
    elif not os.path.isabs(model_path):
        model_path = os.path.join(REPO_ROOT, model_path)

    if cache_path is None:
        cache_path = os.path.join(results_dir, "wild_embeddings.npz")
    elif not os.path.isabs(cache_path):
        cache_path = os.path.join(REPO_ROOT, cache_path)

    if output_csv is None:
        output_csv = os.path.join(results_dir, "wild_predictions.csv")
    elif not os.path.isabs(output_csv):
        output_csv = os.path.join(REPO_ROOT, output_csv)

    # Locate mapping CSV if not provided
    if mapping_path is None:
        candidate_mappings = [
            os.path.join(results_dir, "name_mapping_wild.csv"),
            os.path.join(results_dir, "name_mapping_data_wild_processed.csv"),
            os.path.join(results_dir, "name_mapping.csv"),
        ]
        for cand in candidate_mappings:
            if os.path.isfile(cand):
                mapping_path = cand
                break
        if mapping_path is None:
            mapping_path = candidate_mappings[0]
    elif not os.path.isabs(mapping_path):
        mapping_path = os.path.join(REPO_ROOT, mapping_path)

    print("=" * 78)
    print("TrustLens Wild Dataset Evaluation (CLIP ViT-Large/14 + Logistic Regression)")
    print("=" * 78)
    print(f"Data directory:    {os.path.relpath(data_dir_abs, REPO_ROOT).replace('\\', '/')}")
    print(f"Mapping CSV:       {os.path.relpath(mapping_path, REPO_ROOT).replace('\\', '/')}")
    print(f"Model path:        {os.path.relpath(model_path, REPO_ROOT).replace('\\', '/')}")
    print(f"Output CSV:        {os.path.relpath(output_csv, REPO_ROOT).replace('\\', '/')}\n")

    # Load mapping dictionary
    mapping_dict = load_mapping_dict(mapping_path)
    if mapping_dict:
        print(f"Loaded {len(mapping_dict)} entries from mapping CSV.")
    else:
        print("Note: Mapping CSV not found or empty. Using directory-based subfolder detection.")

    # Discover images
    image_entries, subfolders, metadata = discover_wild_images(data_dir_abs, mapping_dict)
    print(f"Discovered {len(image_entries)} images in {os.path.relpath(data_dir_abs, REPO_ROOT).replace('\\', '/')}.")

    if not image_entries:
        print("\nNo images found to evaluate under data_wild_processed/.")
        print("Ensure you normalize data/wild_test first via:")
        print("  py scripts/prepare_dataset.py --src data/wild_test --dst data_wild_processed")
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "subfolder", "true label", "predicted probability of AI"])
        print(f"Saved empty predictions template to {os.path.relpath(output_csv, REPO_ROOT).replace('\\', '/')}")
        return

    # Load model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Trained classifier not found at {model_path}. Run scripts/train_clip_baseline.py first.")

    print(f"\nLoading trained classifier from {os.path.relpath(model_path, REPO_ROOT).replace('\\', '/')}...")
    clf = joblib.load(model_path)

    # Extract or load embeddings
    embeddings, labels, file_paths = extract_or_load_embeddings(image_entries, cache_path)

    # Predict probabilities of AI (class 1)
    probs = clf.predict_proba(embeddings)[:, 1]
    preds_binary = (probs >= 0.5).astype(int)

    # Save results/wild_predictions.csv
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "subfolder", "true label", "predicted probability of AI"])
        for p, sf, y, prob in zip(file_paths, subfolders, labels, probs):
            writer.writerow([p, sf, int(y), round(float(prob), 5)])

    print(f"\nSaved predictions to {os.path.relpath(output_csv, REPO_ROOT).replace('\\', '/')}\n")

    # Overall metrics
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        auc_score = roc_auc_score(labels, probs)
        auc_str = f"{auc_score:.4f}"
    else:
        auc_str = "N/A (only one class present)"

    acc_score = accuracy_score(labels, preds_binary)
    f1 = f1_score(labels, preds_binary, zero_division=0)

    print("=" * 78)
    print("Overall Performance Metrics (Threshold = 0.5)")
    print("=" * 78)
    print(f"  Overall AUC:           {auc_str}")
    print(f"  Accuracy @ 0.5:        {acc_score:.4f} ({acc_score * 100:.1f}%)")
    print(f"  F1 Score:              {f1:.4f}")
    print(f"  Total Images:          {len(labels)} ({sum(labels == 0)} Real, {sum(labels == 1)} AI)")
    print("=" * 78)

    # Table by subfolder
    subfolder_stats = {}
    for sf, y, prob, pred in zip(subfolders, labels, probs, preds_binary):
        if sf not in subfolder_stats:
            subfolder_stats[sf] = {"count": 0, "probs": [], "correct": 0}
        subfolder_stats[sf]["count"] += 1
        subfolder_stats[sf]["probs"].append(prob)
        if pred == y:
            subfolder_stats[sf]["correct"] += 1

    print("\n" + "=" * 78)
    print("Subfolder Performance Breakdown")
    print("=" * 78)
    print(f"{'Subfolder':<32} {'Count':<8} {'Mean Pred Prob':<18} {'Correct @ 0.5 (%)':<18}")
    print("-" * 78)

    for sf in sorted(subfolder_stats.keys()):
        st = subfolder_stats[sf]
        count = st["count"]
        mean_p = float(np.mean(st["probs"]))
        pct_correct = (st["correct"] / count) * 100.0 if count > 0 else 0.0
        print(f"{sf:<32} {count:<8} {mean_p:<18.4f} {pct_correct:<17.1f}%")

    print("=" * 78)

    # Top 10 most confident wrong predictions
    wrong_indices = [i for i, (y, p_bin) in enumerate(zip(labels, preds_binary)) if y != p_bin]

    print("\n" + "=" * 78)
    print("Top 10 Most Confident Wrong Predictions")
    print("=" * 78)

    if not wrong_indices:
        print("No incorrect predictions! (100% classification accuracy at threshold 0.5)")
    else:
        wrong_entries = []
        for i in wrong_indices:
            y = labels[i]
            p = float(probs[i])
            conf_error = p if y == 0 else (1.0 - p)
            err_type = "False Positive (Real called AI)" if y == 0 else "False Negative (AI called Real)"
            wrong_entries.append({
                "path": file_paths[i],
                "subfolder": subfolders[i],
                "true_label": "Real" if y == 0 else "AI",
                "prob": p,
                "conf_error": conf_error,
                "err_type": err_type,
            })

        wrong_entries.sort(key=lambda x: x["conf_error"], reverse=True)
        top10 = wrong_entries[:10]

        print(f"{'#':<3} {'Path':<45} {'True':<6} {'Pred Prob':<12} {'Error Conf':<12}")
        print("-" * 78)
        for rank, item in enumerate(top10, start=1):
            p_display = item["path"]
            if len(p_display) > 43:
                p_display = "..." + p_display[-40:]
            print(f"{rank:<3} {p_display:<45} {item['true_label']:<6} {item['prob']:<12.4f} {item['conf_error'] * 100:.1f}% ({item['err_type'].split()[0]})")

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="TrustLens Wild Dataset Evaluation")
    parser.add_argument(
        "--data-dir",
        default="data_wild_processed",
        help="Processed wild data directory (default: data_wild_processed)",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Mapping CSV path (default: auto-detected from results/)",
    )
    parser.add_argument(
        "--model",
        default="models/image_clf.joblib",
        help="Classifier model path (default: models/image_clf.joblib)",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Embeddings cache file path (default: results/wild_embeddings.npz)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: results/wild_predictions.csv)",
    )
    args = parser.parse_args()

    evaluate_wild(
        data_dir=args.data_dir,
        mapping_path=args.mapping,
        model_path=args.model,
        cache_path=args.cache,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
