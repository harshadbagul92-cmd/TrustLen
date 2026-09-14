import os
import sys
import tempfile
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import InferenceClient

token = os.getenv("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN is missing from environment.")
    sys.exit(1)

models_to_test = [
    "umm-maybe/AI-image-detector",
    "dima806/ai_vs_real_image_detection",
    "Organika/sdxl-detector",
    "prithivMLmods/Deep-Fake-Detector-v2-Model"
]

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "data", "images"))
test_images = {
    "normal_camera_photo.jpg": os.path.join(data_dir, "normal_camera_photo.jpg"),
    "ai_generated_image.jpg": os.path.join(data_dir, "ai_generated_image.jpg"),
    "deepfake_manipulated_image.jpg": os.path.join(data_dir, "deepfake_manipulated_image.jpg"),
    "compressed_image.jpg": os.path.join(data_dir, "compressed_image.jpg")
}


def test_model_on_image(client, img_path):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        img = Image.open(img_path)
        img.save(temp_path, format="JPEG")

    try:
        res = client.image_classification(temp_path)
        raw_labels = []
        raw_scores = []
        if res and isinstance(res, list):
            for item in res:
                lbl = getattr(item, "label", None) or (item.get("label") if isinstance(item, dict) else None)
                scr = getattr(item, "score", None) or (item.get("score") if isinstance(item, dict) else 0.0)
                raw_labels.append(str(lbl))
                raw_scores.append(round(float(scr), 4))
        return "SUCCESS", raw_labels, raw_scores
    except Exception as e:
        err_msg = str(e)
        if token in err_msg:
            err_msg = err_msg.replace(token, "[TOKEN_REDACTED]")
        return f"FAILED: {err_msg}", [], []
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    print("==================================================")
    print("PHASE 2 -- REAL HF API CANDIDATE EVALUATION")
    print("==================================================")

    model_results = {}

    for model_id in models_to_test:
        print(f"\nEvaluating Model: {model_id}")
        print("-" * 50)
        client = InferenceClient(token=token, model=model_id)

        model_results[model_id] = {}

        for img_name, img_path in test_images.items():
            status, labels, scores = test_model_on_image(client, img_path)
            model_results[model_id][img_name] = {
                "status": status,
                "labels": labels,
                "scores": scores
            }
            print(f"  * {img_name:30s} -> Status: {status}")
            if labels:
                print(f"    Labels: {labels}")
                print(f"    Scores: {scores}")

    print("\n==================================================")
    print("EVALUATION MATRIX SUMMARY")
    print("==================================================")
    print(json.dumps(model_results, indent=2))


if __name__ == "__main__":
    main()
