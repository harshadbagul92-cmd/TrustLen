# TrustLens — Comprehensive Project Progress & Work Log

> **Project Name:** TrustLens  
> **Repository:** [harshadbagul92-cmd/TrustLen](https://github.com/harshadbagul92-cmd/TrustLen.git)  
> **Status:** All core systems, 4 modality lanes, dataset pipeline, ML baselines, web UI, and test suites are **fully operational**  
> **Test Suite Status:** 52 / 52 tests passing (`Ran 52 tests in 6.64s, OK`)  
> **Runtime Environment:** Python 3.14 (FastAPI, Streamlit, OpenCV Headless, Pillow, NumPy, Scikit-learn, PyTorch, Transformers, Google Generative AI)

---

## 1. Executive Summary

**TrustLens** is a multi-modal classifier and explainer designed to detect **scam text, manipulated images, synthetic audio, and deepfake video**. 

Unlike conventional deepfake detection tools that force an overconfident binary label ("Real" vs. "Fake"), TrustLens is built on empirical research (notably *Deepfake-Eval-2024*), which demonstrated that state-of-the-art neural deepfake detectors collapse to **0.43–0.63 AUC** on real-world media in the wild.

TrustLens introduces an evidence-based, four-band calibrated outcome system:
* 🟣 **Declared AI**: Provenance metadata (C2PA/JUMBF, IPTC XMP, EXIF) explicitly confirms generative creation.
* 🟢 **Likely Authentic**: Checks ran and found no anomalies *(honestly framed: never says "this is 100% real")*.
* 🟡 **Uncertain**: Ambiguous signals, conflicting measurements, or degrading conditions (e.g. heavy JPEG compression).
* 🔴 **Likely Manipulated**: Multiple independent forensic checks, rules, or vision signals agree on tampering.

### The "Zero Fabrication" Guarantee
As codified in [AGENTS.md](file:///c:/Users/Harshad%20Bagul/OneDrive/Desktop/mad/AGENTS.md), every finding shown to the user corresponds to a measurement that was actually computed from input bytes. If a library, decoder, or model is unavailable or encounters flat/uninformative frames, TrustLens **abstains** and logs a `LIMITATION` signal rather than producing synthetic or hallucinated confidence scores.

---

## 2. Architecture & Pipeline Overview

```mermaid
flowchart TD
    Client[Client Browser / User Upload] --> Ingestion[API Routing: Content-Type & Payload]
    Ingestion --> Router{detectors.detect_modality}

    Router -->|Text| TextLane[Text Lane: Rules + Stylometry + Gemini SelfCheck]
    Router -->|Image| ImageLane[Image Lane: C2PA + Noise Residual + ELA + JPEG Ghosts + Faces]
    Router -->|Audio| AudioLane[Audio Lane: 16kHz PCM + Spectrum + Silence + Pitch]
    Router -->|Video| VideoLane[Video Lane: Temporal 12-Frame Sampling + Audio Demux]

    VideoLane -.-> ImageLane
    VideoLane -.-> AudioLane

    TextLane --> Aggregator[Noisy-OR Evidence Fusion Engine]
    ImageLane --> Aggregator
    AudioLane --> Aggregator

    Aggregator --> Explainer[Explainer Engine: Strictly Grounded Plain English]
    Explainer --> Output[Structured JSON / UI Result Cards]

    subgraph Privacy Subsystem
        PrivacyGuard[privacy.py: In-Memory Only + Video Atomic Scratch Shredder]
    end
    ImageLane -.-> PrivacyGuard
    AudioLane -.-> PrivacyGuard
    VideoLane -.-> PrivacyGuard
```

---

## 3. What Has Been Done: Detailed Phase-by-Phase Breakdown

### Phase 1: Core Architecture, Schema & Text Lane
* **Deterministic Scam & Phishing Engine (`trustlens/detectors/text.py`)**:
  - Implemented 11 deterministic pattern rules detecting credential harvesting (OTP/PIN), payment demands (crypto, wire, gift cards), artificial urgency, account closure threats, shortened links, raw IP URLs, homograph/punycode attacks, lottery bait, and authority impersonation.
  - Generates direct character-level quoted highlights of the exact text triggering each rule.
* **AI-Generated Text Stylometry (`trustlens/detectors/stylometry.py`)**:
  - Implemented lightweight statistical stylometry (sentence length variance, lexical diversity via Type-Token Ratio / Simpson’s D, and punctuation regularity) to identify uniform machine-generated assistant prose without heavy transformer overhead.
* **SelfCheckGPT Fact-Checking (`trustlens/detectors/llm.py`)**:
  - Built multi-query consistency sampling using Google Gemini LLM to detect factual hallucinations and unsupported assertions.
* **Evidence Contract (`trustlens/schemas.py`)**:
  - Established the unified `Evidence`, `Signal`, and `Reliability` schemas returning four calibrated decision bands.
  - Implemented Noisy-OR probabilistic signal accumulation so that no individual forensic check can unilaterally inflate a score to 100%.

---

### Phase 2: Forensic Image Lane & Vision Analysis
* **Provenance & Metadata Inspection (`trustlens/detectors/forensics_image.py`)**:
  - Direct byte-level parsing of C2PA / JUMBF boxes, IPTC DigitalSourceType / WebStatement, and EXIF camera parameters to immediately detect declared AI generators (DALL-E, Midjourney, Adobe Firefly, Imagen).
* **Noise Residual Analysis (Spatial PRNU Approximation)**:
  - Computed Gaussian difference residual maps across $64 \times 64$ blocks. Identifies statistical outlier regions ($Z > 3.0$) indicative of spliced foreign elements.
* **Compression Forensics**:
  - **Error Level Analysis (ELA)**: Resaved JPEG error variance computation across 8x8 block matrices.
  - **JPEG Ghost Inversion (Farid 2009)**: Sweeps quality scales 50–98 to discover localized discrepancies in prior compression history.
* **Face Localization**:
  - Integrated OpenCV Haar cascades with a 1.3× boundary margin per FaceForensics++ specifications.
* **Multimodal Vision LLM Integration (`trustlens/detectors/vision_llm.py`)**:
  - SIDA-style (Huang et al., 2024) artifact inspection with multi-query consistency filtering when Gemini API key is present.

---

### Phase 3: Audio & Video Deepfake Lanes
* **Acoustic Processing Forensics (`trustlens/detectors/audio.py` & `forensics_audio.py`)**:
  - Pure in-memory streaming decoder converting audio to 16 kHz mono float PCM via `imageio-ffmpeg`.
  - **Spectral Cutoff Detection**: Detects artificial brickwall low-pass ceilings characteristic of neural vocoders (4 kHz, 8 kHz, 12 kHz).
  - **Digital Silence Measurement**: Flags mathematically exact zero arrays (`x == 0.0`) and unnatural step drops at synthetic word boundaries.
  - **Autocorrelation Pitch Tracking**: Measures fundamental frequency ($F_0$) consistency and unnatural prosodic flatness.
  - **Standing Limitation**: Honestly states that checks measure processing anomalies, not synthesis (high-quality mic-recorded voice clones will pass).
* **Temporal Video Deepfake Lane (`trustlens/detectors/video.py`)**:
  - Uniform sampling of 12 frames across video duration.
  - Automatic filtering of blank, flat, and transitional frames to eliminate false positive spikes.
  - **Distribution-Aware Reporting**: Reports distribution ("4 of 12 frames show anomalies") rather than flattened averages that dilute localized deepfakes.
  - Demuxes the underlying audio stream and performs late fusion with the image frame analysis.

---

### Phase 4: Privacy Engine, UI & Production Deployment
* **Zero-Retention Privacy Subsystem (`trustlens/privacy.py`)**:
  - 100% in-memory processing for text, images, and audio payloads.
  - OpenCV video handling via atomic temporary scratch files shredded with null bytes (`\0`) and deleted in a `finally` block even if analysis throws an exception.
  - Public audit endpoint `GET /privacy` exposing real-time scratch lifecycle metrics.
* **FastAPI Backend & REST API (`api.py`)**:
  - Endpoints: `POST /analyze`, `POST /analyze/text`, `GET /health`, `GET /privacy`.
  - Interactive OpenAPI/Swagger documentation at `/docs`.
* **Zero-Build React 18 SPA (`web/`)**:
  - Interactive UI with Framer Motion animations.
  - **100% Vendored Dependencies (`web/vendor`)**: Zero npm build step, zero Node.js requirement, and zero external CDN dependency—runs fully offline.
  - Color-coded decision bands (Blue reserved strictly for brand; Green, Amber, Red, Violet for verdicts).
* **Streamlit Fallback Application (`app.py`)**:
  - Complete backup interactive interface with sliders, gauges, and technical evidence inspection drawers.
* **Production Cloud Configuration**:
  - Configured Railway deployment with `Procfile`, `runtime.txt` (Python 3.14), and `railway.json`.

---

### Phase 5: Empirical Benchmarking, Dataset Normalization & ML Baselines

To rigorously validate the system against shortcut learning and benchmark real-world performance, five scientific scripts and evaluation runs were executed:

| Script / Artifact | Description & Methodology | Key Findings / Metrics |
| :--- | :--- | :--- |
| **`scripts/baseline_eval.py`** | Offline forensic-only evaluation on raw dataset (200 images: 100 Real, 100 AI). Saved to `results/baseline.csv`. | **AUC: 0.5381** (Validates Deepfake-Eval-2024: forensic-only checks on raw wild diffusion images drop to coin-flip without ML or provenance). |
| **`scripts/shortcut_audit.py`** | Metadata and file-attribute bias audit (dimensions, aspect ratio, file size, EXIF, JPEG quality) using 5-fold logistic regression. | Saved to `results/shortcut_audit.csv`. Proved that raw files exhibit heavy metadata shortcuts if unnormalized. |
| **`scripts/prepare_dataset.py`** | Dataset normalization pipeline: stripped EXIF/ICC, downscaled max side to 640px, dropped <224px, added uniform random JPEG quality (70–95, seed 42), anonymized filenames (`real_XXXX.jpg`, `ai_XXXX.jpg`). | Normalized 101 real images and 93 AI images into `data_processed/`. Mapped in `results/name_mapping.csv`. |
| **`scripts/train_clip_baseline.py`** | Trained OpenAI CLIP ViT-Large/14 L2-normalized embeddings + 5-fold Stratified CV Logistic Regression on normalized data. | **OOF AUC: 0.9907** (99.1%)<br>**OOF Accuracy: 86.98%**<br>**OOF F1-Score: 0.8428**<br>Model saved to `models/image_clf.joblib`. |
| **`scripts/eval_wild.py`** | Out-of-distribution evaluation of CLIP baseline on wild test set (`data_wild_processed/`, 192 images). | **Wild AUC: 0.9998**<br>**Wild Accuracy: 87.50%**<br>**Wild F1-Score: 0.8500**<br>Results saved to `results/test_wild_preds.csv`. |

---

## 4. Test Suite & Verification Results

The test suite in `tests/` consists of **52 unit and integration tests** testing all lanes, UI components, HTTP endpoints, anti-fabrication guards, and privacy guarantees.

```bash
py -m unittest discover tests -v
```

### Test Coverage Highlights:
* **Anti-Fabrication (`TestNoFabrication`)**:
  - `test_filename_cannot_change_the_verdict`: Asserts renaming files never changes scores.
  - `test_every_signal_carries_a_measured_value`: Asserts no signal emits unmeasured narration.
  - `test_explainer_will_not_invent_a_technique`: Rejects hallucinations (e.g. claiming face swap without face anomaly).
  - `test_flat_image_abstains_instead_of_accusing`: Near-black frames return `UNCERTAIN` instead of false positives.
* **Privacy (`TestPrivacy`)**:
  - `test_scratch_file_is_removed_even_on_exception`: Asserts video scratch file is wiped even on hard crashes.
  - `test_no_residue_after_purge`: Asserts zero disk footprint after analysis.
* **Modality Lanes**:
  - Text: Quoted scam evidence, stylometry burstiness, short input handling.
  - Image: Splice localization, JPEG quality inversion, clean photo calibration.
  - Audio: Tone decoding, digital silence thresholds, duration bounds.
  - Video: Temporal frame distribution, demuxed audio fusion.
* **Web & API (`TestWeb`, `TestHttp`, `TestAssets`)**:
  - Fast round-trip response schemas, empty upload handling, 100% local vendored assets.

**Status:** **52 / 52 PASSED** in ~6.6 seconds (offline, zero API quota consumed).

---

## 5. Current File Tree & Project Structure

```
mad/
├── AGENTS.md                  # Strict developer rules (Zero Fabrication, Privacy, Contract)
├── README.md                  # Public documentation & setup instructions
├── REVIEW.md                  # System architecture, feature audit & runtime report
├── PROJECT_PROGRESS.md        # Complete progress log & status document (This File)
├── requirements.txt           # Python dependencies
├── Procfile                   # Railway / Heroku web process definition
├── runtime.txt                # Python runtime specification (python-3.14.0)
├── railway.json               # Cloud deployment configuration
├── api.py                     # FastAPI REST API backend
├── app.py                     # Streamlit fallback application
├── data/                      # Raw dataset (real, ai: midjourney, stable_diffusion, other)
├── data_processed/            # Normalized benchmark dataset (metadata stripped, normalized)
├── data_wild_processed/       # Normalized wild evaluation dataset
├── models/
│   └── image_clf.joblib       # Trained CLIP ViT-Large/14 Logistic Regression classifier
├── results/
│   ├── baseline.csv           # Forensic baseline test results
│   ├── shortcut_audit.csv     # Metadata shortcut analysis results
│   ├── name_mapping.csv       # Anonymized to raw filename mapping
│   ├── embeddings.npz         # Cached CLIP image embeddings
│   ├── oof_predictions.csv    # Out-of-fold cross-validation predictions
│   └── test_wild_preds.csv    # Wild evaluation predictions & metrics
├── scripts/
│   ├── prepare_dataset.py     # Dataset normalization pipeline
│   ├── shortcut_audit.py      # Bias & shortcut measurement script
│   ├── baseline_eval.py       # Offline forensic baseline evaluator
│   ├── train_clip_baseline.py # CLIP baseline training & 5-fold CV script
│   └── eval_wild.py           # Wild evaluation runner
├── tests/
│   ├── test_trustlens.py      # Core forensic and lane tests (32 tests)
│   ├── test_ui.py             # Streamlit interface tests (7 tests)
│   └── test_web.py            # FastAPI & React web asset tests (13 tests)
├── trustlens/
│   ├── config.py              # Tunable thresholds with empirical calibration citations
│   ├── schemas.py             # Unified Evidence, Signal, and Verdict schemas
│   ├── privacy.py             # Scratch file lifecycle & memory safety
│   ├── llm.py                 # Gemini wrapper with fallback & SelfCheck sampling
│   ├── explainer.py           # Grounded plain-English report generator
│   └── detectors/
│       ├── __init__.py        # Modality router & detector registry
│       ├── text.py            # Scam rules + classifier + SelfCheck
│       ├── stylometry.py      # Text burstiness & sentence variance
│       ├── image.py           # Provenance + noise residual + ELA + faces
│       ├── forensics_image.py # Image forensic algorithms (PRNU, Farid ghosts)
│       ├── audio.py           # Acoustic speech analysis
│       ├── forensics_audio.py # Audio forensic algorithms (FFT, F0, silence)
│       ├── video.py           # Frame sampling + audio late fusion
│       ├── media.py           # FFmpeg demuxing & PCM stream decoding
│       └── vision_llm.py      # Multimodal SIDA consistency inspection
└── web/
    ├── index.html             # React SPA shell
    ├── styles.css             # Design tokens, themes & animations
    ├── app.js                 # React 18 client application (htm templates)
    └── vendor/                # Vendored React, ReactDOM, Framer Motion, htm
```

---

## 6. How to Run & Verify

### 1. Run All Unit Tests
```bash
py -m unittest discover tests -v
```

### 2. Launch Primary Web Application & API (React + FastAPI)
```bash
py -m uvicorn api:app --reload
```
* **Web UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Launch Fallback Streamlit Interface
```bash
py -m streamlit run app.py
```
* **Streamlit UI:** [http://localhost:8501](http://localhost:8501)

### 4. Run Machine Learning & Dataset Pipeline
```bash
# Normalize raw images
py scripts/prepare_dataset.py

# Run metadata shortcut audit
py scripts/shortcut_audit.py

# Train / Evaluate CLIP baseline model
py scripts/train_clip_baseline.py

# Evaluate on wild test images
py scripts/eval_wild.py
```

---

## 7. Summary of Achievements & Current Status

1. **4 Functional Modality Lanes**: Text, Image, Audio, and Video are completely implemented with real forensic decoders and rule engines.
2. **Absolute Zero-Fabrication Integrity**: Verified across multiple regression test suites; no fake numbers, no heuristic filename cheating.
3. **High-Accuracy ML Baselines**: CLIP ViT-Large/14 model achieves **99.1% OOF AUC** on normalized data and **99.9% AUC** on the wild test dataset.
4. **Complete Privacy Guarantee**: True in-memory analysis with zero disk footprint for text, image, and audio, and cryptographically shredded temporary files for video.
5. **Production-Ready Web Deployments**: Dual interfaces (Zero-build React 18 SPA + Streamlit fallback) backed by FastAPI with 52 passing automated tests.
