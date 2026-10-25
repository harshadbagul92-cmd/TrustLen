"""
Stream and download a Hugging Face image dataset for TrustLens.

Features:
1. Streams data using load_dataset(..., streaming=True) and .take(limit)
   so the full dataset is never downloaded or cached on disk.
2. Dynamically detects label types (string 'real'/'fake' or ClassLabel / int).
3. Saves real images to data/real/ and fake images to data/ai/other/.
4. Converts all saved images to standard RGB JPEG format.
5. Uses filenames like hf_nanobananapro_0001.jpg and never overwrites existing files.
6. Reports total images saved and new disk space consumed in MB.
"""
import argparse
import io
import os
import sys
from typing import Optional, Tuple
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def resolve_label(
    raw_label: any, class_names: Optional[list]
) -> Tuple[str, str]:
    """
    Determine if a raw label represents 'real' or 'fake'/AI.
    Returns (category, description) where category is 'real' or 'fake'.
    """
    # 1. If raw_label is an integer
    if isinstance(raw_label, int) or (isinstance(raw_label, str) and raw_label.isdigit()):
        int_val = int(raw_label)
        if class_names and 0 <= int_val < len(class_names):
            name = str(class_names[int_val]).strip().lower()
            desc = f"int {int_val} mapped via ClassLabel to '{name}'"
            if "real" in name or "authentic" in name:
                return "real", desc
            if "fake" in name or "ai" in name or "synth" in name:
                return "fake", desc
        # Default convention if no ClassLabel metadata: 0=fake, 1=real (as in julienlucas dataset)
        if int_val == 1:
            return "real", f"int {int_val} (interpreted as real)"
        return "fake", f"int {int_val} (interpreted as fake/AI)"

    # 2. If raw_label is a string
    if isinstance(raw_label, str):
        label_str = raw_label.strip().lower()
        if "real" in label_str or "authentic" in label_str:
            return "real", f"string '{raw_label}' (detected as real)"
        if "fake" in label_str or "ai" in label_str or "synth" in label_str:
            return "fake", f"string '{raw_label}' (detected as fake/AI)"
        return "fake", f"string '{raw_label}' (unrecognized, mapped to fake/AI)"

    # 3. Fallback for unexpected formats
    return "fake", f"unknown type {type(raw_label).__name__} (value: {raw_label})"


def get_next_filename(target_dir: str, prefix: str, start_index: int) -> Tuple[str, str, int]:
    """
    Find the next available non-conflicting filename under target_dir.
    Guarantees no existing files are ever overwritten.
    Returns (filename, full_path, next_index).
    """
    idx = start_index
    while True:
        filename = f"{prefix}_{idx:04d}.jpg"
        full_path = os.path.join(target_dir, filename)
        if not os.path.exists(full_path):
            return filename, full_path, idx + 1
        idx += 1


def main():
    parser = argparse.ArgumentParser(
        description="Stream and download image datasets from Hugging Face for TrustLens."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="julienlucas/midjourney-dalle-sd-nanobananapro-dataset",
        help="Hugging Face dataset identifier (default: julienlucas/midjourney-dalle-sd-nanobananapro-dataset)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of images to download total (default: 1000)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to stream (default: train)",
    )
    args = parser.parse_args()

    # Destination directories
    real_dir = os.path.join(REPO_ROOT, "data", "real")
    fake_dir = os.path.join(REPO_ROOT, "data", "ai", "other")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Naming prefix
    if "nanobananapro" in args.dataset.lower():
        prefix = "hf_nanobananapro"
    else:
        clean_name = args.dataset.split("/")[-1].replace("-", "_").lower()
        prefix = f"hf_{clean_name}"

    print("=" * 70)
    print("TrustLens — Hugging Face Streaming Dataset Downloader")
    print("=" * 70)
    print(f"Dataset:       {args.dataset}")
    print(f"Split:         {args.split}")
    print(f"Limit:         {args.limit} images")
    print(f"Real target:   {real_dir}")
    print(f"Fake target:   {fake_dir}")
    print(f"File prefix:   {prefix}_XXXX.jpg")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n[ERROR] 'datasets' library is not installed. Please install it with:")
        print("    pip install datasets")
        sys.exit(1)

    print("\nConnecting to Hugging Face Hub (streaming=True)...")
    try:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=6000)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset '{args.dataset}': {e}")
        sys.exit(1)

    # Inspect dataset features for ClassLabel mapping
    class_names = None
    try:
        if hasattr(dataset, "features") and "label" in dataset.features:
            label_feature = dataset.features["label"]
            if hasattr(label_feature, "names"):
                class_names = label_feature.names
                print(f"[Schema] Detected ClassLabel mapping: {class_names}")
    except Exception:
        pass

    # Use .take(limit) on the streaming dataset
    stream_iter = dataset.take(args.limit)

    saved_real = 0
    saved_fake = 0
    bytes_written = 0
    file_counter = 1

    print("\nStarting streaming download...")
    for item_idx, item in enumerate(stream_iter):
        raw_label = item.get("label")
        category, desc = resolve_label(raw_label, class_names)

        # Select target directory
        target_dir = real_dir if category == "real" else fake_dir
        filename, dest_path, file_counter = get_next_filename(target_dir, prefix, file_counter)

        # Extract and convert image to RGB
        raw_img = item.get("image")
        if raw_img is None:
            print(f"[{item_idx + 1}/{args.limit}] Skipped item {item_idx}: 'image' field is None.")
            continue

        try:
            if isinstance(raw_img, Image.Image):
                img = raw_img
            elif isinstance(raw_img, dict) and "bytes" in raw_img and raw_img["bytes"]:
                img = Image.open(io.BytesIO(raw_img["bytes"]))
            else:
                img = Image.open(raw_img)

            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save as JPEG without overwriting existing files
            img.save(dest_path, format="JPEG", quality=95)
            file_size = os.path.getsize(dest_path)
            bytes_written += file_size

            if category == "real":
                saved_real += 1
            else:
                saved_fake += 1

            print(
                f"[{item_idx + 1}/{args.limit}] Label: {raw_label} ({desc}) -> Saved {category} to {filename} ({file_size / 1024:.1f} KB)"
            )

        except Exception as e:
            print(f"[{item_idx + 1}/{args.limit}] Error saving image {item_idx}: {e}")

    # Summary
    total_saved = saved_real + saved_fake
    total_mb = bytes_written / (1024 * 1024)

    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Total images requested:     {args.limit}")
    print(f"Total images saved:         {total_saved}")
    print(f"  - Real images:            {saved_real} -> data/real/")
    print(f"  - Fake / AI images:       {saved_fake} -> data/ai/other/")
    print(f"Approximate disk space:     {total_mb:.2f} MB ({bytes_written:,} bytes)")
    print("=" * 70)
    print("[Done] Existing files in data/ were preserved without modification.")


if __name__ == "__main__":
    main()
