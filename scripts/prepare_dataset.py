"""
Dataset preparation script for TrustLens.

Creates a normalized copy of the dataset under data_processed/ so that
real and AI images cannot be told apart by file format, size, metadata,
or JPEG quality. Original files in data/ are never modified.

Rules:
1. Clean destination folder completely before processing.
2. Open with Pillow, convert to RGB, drop all metadata (EXIF, ICC, etc.).
3. Center-crop every image to a square using the shorter side.
4. Skip images whose shorter side is under 224 px, and print skipped images.
5. Always resize the square crop to exactly 640x640 px (LANCZOS).
6. Save as JPEG with random quality drawn uniformly from 70 to 95 using a
   fixed random seed for reproducibility.
7. Keep folder structure under data_processed/real/ and data_processed/ai/,
   and use neutral file names (real_0001.jpg, ai_0001.jpg). Save mapping
   to results/name_mapping.csv.
8. Print counts kept and skipped per class.
"""
import os
import sys
import csv
import time
import random
import shutil
from typing import List, Tuple
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RANDOM_SEED = 42
TARGET_SIDE = 640
TARGET_MAX_SIDE = TARGET_SIDE
MIN_SHORT_SIDE = 224
QUALITY_MIN = 70
QUALITY_MAX = 95


def find_images(base_dir: str, label: int) -> List[Tuple[str, int, str]]:
    """
    Find images recursively.
    Returns (full_path, label, rel_path_under_base_dir).
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
                items.append((full_path, label, rel_path))

    items.sort(key=lambda x: x[0])
    return items


def get_mapping_path(results_dir: str, src_dir: str, dest_dir: str, custom_mapping: str = None) -> str:
    """Determine mapping CSV path based on source and destination directories."""
    if custom_mapping:
        return custom_mapping if os.path.isabs(custom_mapping) else os.path.join(results_dir, custom_mapping)

    rel_src = os.path.relpath(src_dir, REPO_ROOT).replace("\\", "/")
    rel_dst = os.path.relpath(dest_dir, REPO_ROOT).replace("\\", "/")

    # Default run: results/name_mapping.csv
    if rel_src == "data" and rel_dst == "data_processed":
        return os.path.join(results_dir, "name_mapping.csv")

    dst_tag = os.path.basename(os.path.normpath(dest_dir))
    if "wild" in dst_tag.lower() or "wild" in rel_src.lower():
        return os.path.join(results_dir, "name_mapping_wild.csv")
    return os.path.join(results_dir, f"name_mapping_{dst_tag}.csv")


def prepare_dataset(src: str = "data", dst: str = "data_processed", mapping: str = None):
    src_data_dir = src if os.path.isabs(src) else os.path.join(REPO_ROOT, src)
    dest_data_dir = dst if os.path.isabs(dst) else os.path.join(REPO_ROOT, dst)
    results_dir = os.path.join(REPO_ROOT, "results")
    mapping_csv_path = get_mapping_path(results_dir, src_data_dir, dest_data_dir, mapping)

    rel_src_display = os.path.relpath(src_data_dir, REPO_ROOT).replace("\\", "/")
    rel_dst_display = os.path.relpath(dest_data_dir, REPO_ROOT).replace("\\", "/")

    # 1. Clean destination folder first
    removed_count = 0
    if os.path.isdir(dest_data_dir):
        for root, _, files in os.walk(dest_data_dir):
            removed_count += len(files)

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                if os.path.exists(dest_data_dir):
                    shutil.rmtree(dest_data_dir)
                break
            except PermissionError:
                if attempt < max_attempts:
                    print(f"Folder locked, retrying in 2s... (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(2)
                else:
                    raise PermissionError(
                        f"Could not delete {dest_data_dir}. Close any File Explorer "
                        "windows, image viewers, or IDE tabs showing files in this folder, "
                        "pause OneDrive syncing, and try again."
                    )

    os.makedirs(dest_data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    real_images = find_images(os.path.join(src_data_dir, "real"), label=0)
    ai_images = find_images(os.path.join(src_data_dir, "ai"), label=1)

    print("=" * 68)
    print("TrustLens Dataset Normalization & Preparation")
    print("=" * 68)
    print(f"Source dir:        {rel_src_display}")
    print(f"Destination dir:   {rel_dst_display}")
    if removed_count > 0:
        print(f"Destination clean: removed {removed_count} old file(s)")
    else:
        print(f"Destination clean: 0 old files removed")
    print(f"Source images:     {len(real_images)} real, {len(ai_images)} AI")
    print(f"Random seed:       {RANDOM_SEED}")
    print(f"Target size:       {TARGET_SIDE}x{TARGET_SIDE} px (exact square) | Min side: {MIN_SHORT_SIDE} px")
    print(f"JPEG quality:      [{QUALITY_MIN}, {QUALITY_MAX}]\n")

    # Reproducible RNG for quality assignment
    rng = random.Random(RANDOM_SEED)

    mapping_rows = []
    stats = {
        0: {"kept": 0, "skipped": 0, "skipped_files": []},
        1: {"kept": 0, "skipped": 0, "skipped_files": []},
    }

    # Process each class
    for label, prefix, image_list in [(0, "real", real_images), (1, "ai", ai_images)]:
        counter = 1
        for orig_path, _, rel_path in image_list:
            display_orig_path = os.path.relpath(orig_path, REPO_ROOT).replace("\\", "/")

            try:
                with Image.open(orig_path) as img:
                    orig_w, orig_h = img.size
                    crop_side = min(orig_w, orig_h)

                    # 3. Skip images whose shorter side is under 224 px
                    if crop_side < MIN_SHORT_SIDE:
                        stats[label]["skipped"] += 1
                        stats[label]["skipped_files"].append(
                            (display_orig_path, f"{orig_w}x{orig_h} (shorter side {crop_side} < {MIN_SHORT_SIDE})")
                        )
                        continue

                    # 1. Convert to RGB, drop all metadata
                    rgb_img = img.convert("RGB")

                    # 2. Center-crop every image to a square using the shorter side
                    left = (orig_w - crop_side) // 2
                    top = (orig_h - crop_side) // 2
                    right = left + crop_side
                    bottom = top + crop_side
                    square_img = rgb_img.crop((left, top, right, bottom))

                    # Always resize the square crop to exactly 640x640 (LANCZOS)
                    final_side = TARGET_SIDE
                    if square_img.size != (final_side, final_side):
                        square_img = square_img.resize((final_side, final_side), Image.Resampling.LANCZOS)

                    # 4. Random quality drawn uniformly from 70 to 95
                    q = rng.randint(QUALITY_MIN, QUALITY_MAX)

                    # 5. Keep folder structure under dest_data_dir/<prefix>/
                    # Neutral file name: real_0001.jpg, ai_0001.jpg
                    rel_dir = os.path.dirname(rel_path)
                    dest_dir = os.path.join(dest_data_dir, prefix, rel_dir)
                    os.makedirs(dest_dir, exist_ok=True)

                    new_filename = f"{prefix}_{counter:04d}.jpg"
                    dest_filepath = os.path.join(dest_dir, new_filename)

                    # Save as clean JPEG (no EXIF or ICC attached)
                    square_img.save(dest_filepath, "JPEG", quality=q)

                    rel_dest_path = os.path.relpath(dest_filepath, REPO_ROOT).replace("\\", "/")
                    mapping_rows.append({
                        "new name": new_filename,
                        "new path": rel_dest_path,
                        "original path": display_orig_path,
                        "label": label,
                        "quality": q,
                        "resolution": f"{final_side}x{final_side}",
                    })

                    stats[label]["kept"] += 1
                    counter += 1

            except Exception as exc:
                stats[label]["skipped"] += 1
                stats[label]["skipped_files"].append((display_orig_path, f"Error: {exc}"))

    # Save mapping CSV
    mapping_fieldnames = ["new name", "new path", "original path", "label", "quality", "resolution"]
    with open(mapping_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mapping_fieldnames)
        writer.writeheader()
        writer.writerows(mapping_rows)

    print(f"Saved file mapping to {os.path.relpath(mapping_csv_path, REPO_ROOT).replace('\\', '/')}\n")

    # 6. Print summary
    print("-" * 68)
    print("Dataset Preparation Summary:")
    print("-" * 68)
    print(f"Real images (label 0):")
    print(f"  Kept:    {stats[0]['kept']}")
    print(f"  Skipped: {stats[0]['skipped']}")
    if stats[0]["skipped_files"]:
        print("  Skipped files:")
        for path, reason in stats[0]["skipped_files"]:
            print(f"    - {path}: {reason}")

    print(f"\nAI images (label 1):")
    print(f"  Kept:    {stats[1]['kept']}")
    print(f"  Skipped: {stats[1]['skipped']}")
    if stats[1]["skipped_files"]:
        print("  Skipped files:")
        for path, reason in stats[1]["skipped_files"]:
            print(f"    - {path}: {reason}")

    print("\n" + "=" * 68)
    print(f"Successfully created normalized dataset in {rel_dst_display}/")
    print(f"Total normalized images: {len(mapping_rows)} ({stats[0]['kept']} real, {stats[1]['kept']} AI)")
    print("=" * 68)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TrustLens Dataset Normalization & Preparation")
    parser.add_argument(
        "--src",
        default="data",
        help="Source directory containing real/ and ai/ subdirectories (default: data)",
    )
    parser.add_argument(
        "--dst",
        default="data_processed",
        help="Destination directory for normalized images (default: data_processed)",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Optional custom mapping CSV path (default: name_mapping.csv or name_mapping_<dst>.csv)",
    )
    args = parser.parse_args()

    prepare_dataset(src=args.src, dst=args.dst, mapping=args.mapping)


if __name__ == "__main__":
    main()
