import os
import sys
import json
from PIL import Image

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from main import app
from backend.config import settings

client = TestClient(app)


def prepare_test_images():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "data", "images"))
    os.makedirs(data_dir, exist_ok=True)

    samples = {
        "normal_camera_photo.jpg": ((100, 150, 200), 512, 512, 95),
        "ai_generated_image.jpg": ((255, 105, 180), 512, 512, 95),
        "deepfake_manipulated_image.jpg": ((220, 20, 60), 512, 512, 95),
        "compressed_image.jpg": ((120, 120, 120), 64, 64, 20)  # Heavy compression < 15KB
    }

    image_paths = {}
    for filename, (color, width, height, quality) in samples.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            img = Image.new("RGB", (width, height), color=color)
            img.save(path, format="JPEG", quality=quality)
        image_paths[filename] = path

    return image_paths


def run_manual_test():
    print("==================================================")
    print("PHASE 2 -- REAL MANUAL API TEST (POST /analyze/image)")
    print("==================================================")
    print(f"HF_TOKEN present:      {'YES' if settings.HF_TOKEN else 'NO'}")
    print(f"Model Name:            {settings.HF_IMAGE_MODEL}")
    print(f"Provider:              {settings.HF_PROVIDER}")
    print("--------------------------------------------------\n")

    images = prepare_test_images()
    results = []

    for filename, filepath in images.items():
        print(f"> Testing Image File: {filename}")
        with open(filepath, "rb") as f:
            file_bytes = f.read()

        response = client.post(
            "/analyze/image",
            files={"file": (filename, file_bytes, "image/jpeg")}
        )

        status_code = response.status_code
        print(f"  * HTTP Status:           {status_code}")

        if status_code == 200:
            payload = response.json()
            res = payload.get("result", {})
            model_info = res.get("model", {})

            item_report = {
                "filename": filename,
                "http_status": status_code,
                "model": model_info.get("name"),
                "provider": model_info.get("provider"),
                "raw_labels": model_info.get("raw_labels", []),
                "raw_scores": model_info.get("raw_scores", []),
                "normalized_prediction": res.get("prediction"),
                "detector_score": res.get("detector_score"),
                "application_confidence": res.get("confidence"),
                "confidence_level": res.get("confidence_level"),
                "evidence": [e.get("description") for e in res.get("evidence", [])],
                "warnings": res.get("warnings", []),
                "explanation": res.get("explanation")
            }

            print(f"  * Model Used:            {item_report['model']}")
            print(f"  * Raw Labels:            {item_report['raw_labels']}")
            print(f"  * Raw Scores:            {item_report['raw_scores']}")
            print(f"  * Normalized Prediction:  {item_report['normalized_prediction']}")
            print(f"  * Detector Score:        {item_report['detector_score']}")
            print(f"  * Application Confidence: {item_report['application_confidence']} ({item_report['confidence_level']})")
            print(f"  * Evidence Count:        {len(item_report['evidence'])}")
            print(f"  * Explanation:          {item_report['explanation']}")
            results.append(item_report)
        else:
            print(f"  * Error Response:        {response.text}")
            results.append({
                "filename": filename,
                "http_status": status_code,
                "error": response.text
            })

        print("-" * 50)

    print("\n==================================================")
    print("MANUAL TEST COMPLETED SUCCESSFULLY")
    print("==================================================")


if __name__ == "__main__":
    run_manual_test()
