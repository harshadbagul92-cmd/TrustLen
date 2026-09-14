import os
import sys
import io
from PIL import Image
from dotenv import load_dotenv

# Load .env file explicitly BEFORE backend imports
load_dotenv()

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import settings, Settings
from backend.schemas.detector_result import PredictionEnum


def ensure_test_images():
    """Ensures 3 representative sample images exist in tests/data/images/."""
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "data", "images"))
    os.makedirs(data_dir, exist_ok=True)

    samples = {
        "normal_camera_photo.jpg": (100, 150, 200),
        "ai_generated_image.jpg": (255, 105, 180),
        "deepfake_manipulated_image.jpg": (220, 20, 60)
    }

    image_paths = {}
    for filename, color in samples.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            img = Image.new("RGB", (512, 512), color=color)
            img.save(path, format="JPEG", quality=95)
        image_paths[filename] = path

    return image_paths


def main():
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    env_exists = os.path.exists(dotenv_path)

    # Force re-read of dotenv to ensure latest disk state
    load_dotenv(dotenv_path, override=True)
    settings = Settings()

    hf_token = settings.HF_TOKEN or os.getenv("HF_TOKEN", "")
    hf_model = settings.HF_IMAGE_MODEL
    hf_provider = settings.HF_PROVIDER

    token_loaded = bool(hf_token and len(hf_token.strip()) > 0)
    token_prefix = f"{hf_token[:3]}*** (length: {len(hf_token)})" if token_loaded else "NONE"

    print("==================================================")
    print("REAL HUGGING FACE API VALIDATION (NO MOCK)")
    print("==================================================")
    print(f".env File Exists:       {'YES' if env_exists else 'NO'} ({dotenv_path})")
    print(f"HF_TOKEN loaded:        {'YES' if token_loaded else 'NO'}")
    print(f"HF_TOKEN prefix:        {token_prefix}")
    print(f"Model Name:            {hf_model}")
    print(f"HF Provider:           {hf_provider}")
    print("--------------------------------------------------\n")

    test_images = ensure_test_images()

    real_api_working = False
    api_error_detail = None
    all_success = True

    try:
        from huggingface_hub import InferenceClient
        # Create InferenceClient with token and model
        client = InferenceClient(token=hf_token if hf_token else None, model=hf_model)
    except Exception as init_err:
        client = None
        api_error_detail = f"Failed to initialize InferenceClient: {str(init_err)}"

    for label_category, filepath in test_images.items():
        filename = os.path.basename(filepath)
        print(f"> Processing Image: {filename}")

        with open(filepath, "rb") as f:
            img_bytes = f.read()

        raw_labels = []
        raw_scores = []
        detector_score = 0.0
        norm_pred = PredictionEnum.UNCERTAIN

        if client:
            try:
                # Perform REAL API call without mock fallback passing filepath so Content-Type header is attached
                raw_response = client.image_classification(filepath)
                real_api_working = True

                if raw_response and isinstance(raw_response, list):
                    for item in raw_response:
                        # Extract label and score whether dict or object
                        lbl = getattr(item, "label", None) or (item.get("label") if isinstance(item, dict) else None)
                        scr = getattr(item, "score", None) or (item.get("score") if isinstance(item, dict) else 0.0)
                        raw_labels.append(str(lbl))
                        raw_scores.append(float(scr))
                    
                    if raw_labels and raw_scores:
                        top_label = raw_labels[0]
                        detector_score = raw_scores[0]
                        clean_lbl = top_label.strip().lower()
                        
                        # Normalize raw label
                        mapped = settings.IMAGE_MODEL_LABEL_MAP.get(clean_lbl)
                        if mapped == "LIKELY_SYNTHETIC":
                            norm_pred = PredictionEnum.LIKELY_SYNTHETIC
                        elif mapped == "LIKELY_AUTHENTIC":
                            norm_pred = PredictionEnum.LIKELY_AUTHENTIC
                        else:
                            norm_pred = PredictionEnum.UNCERTAIN

                print(f"  * API Connection Status: REAL API HTTP 200 OK")
                print(f"  * Model Name:            {hf_model}")
                print(f"  * Raw Labels:            {raw_labels}")
                print(f"  * Raw Scores:            {raw_scores}")
                print(f"  * Normalized Prediction:  {norm_pred.value}")
                print(f"  * Detector Score:        {detector_score:.4f}")

            except Exception as api_err:
                all_success = False
                # Mask secret token from any raw exception string
                err_msg = str(api_err)
                if hf_token and hf_token in err_msg:
                    err_msg = err_msg.replace(hf_token, "[TOKEN_REDACTED]")
                
                api_error_detail = f"Hugging Face Inference API Error: {err_msg}"
                print(f"  * API Connection Status: FAILED (Authentication/Inference Error)")
                print(f"  * Model Name:            {hf_model}")
                print(f"  * Raw Labels:            N/A")
                print(f"  * Raw Scores:            N/A")
                print(f"  * Normalized Prediction:  N/A")
                print(f"  * Detector Score:        N/A")
                print(f"  * Error Message:         {err_msg}")

        print("-" * 50)

    print("\n==================================================")
    print("REAL API VALIDATION SUMMARY")
    print("==================================================")
    print(f"RESULT FROM REAL HF API: {'YES' if (real_api_working and all_success) else 'NO (Failed with Error)'}")
    print(f"MODEL USED:              {hf_model}")
    print(f"API CONNECTION STATUS:   {'SUCCESSFUL' if (real_api_working and all_success) else 'FAILED'}")
    if api_error_detail:
        print(f"ERROR DIAGNOSIS:         {api_error_detail}")
    print("==================================================")


if __name__ == "__main__":
    main()

