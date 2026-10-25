# TrustLens — Comprehensive Codebase Review & Architecture Report

> **Document Type:** System Architecture, Feature Audit & Runtime Verification  
> **Repository:** [TrustLen](https://github.com/harshadbagul92-cmd/TrustLen.git)  
> **Test Suite Status:** 52 / 52 unit tests passing (`Ran 52 tests in 6.493s, OK`)  
> **Environment:** Python 3.14 (with `opencv-python-headless`, `fastapi`, `streamlit`, `imageio-ffmpeg`, `pillow`, `pydantic`, `google-generativeai`)

---

## 1. What is TrustLens?

**TrustLens** is an open-source classifier and explainer designed to detect **scam text, phishing, manipulated images, cloned audio, and deepfake video**.

Unlike conventional deepfake tools that force an overconfident binary "Real" or "Fake" label, TrustLens is built on empirical research (notably *Deepfake-Eval-2024*), which demonstrated that state-of-the-art neural deepfake detectors plummet to 0.43–0.63 AUC on real-world media in the wild. TrustLens explicitly treats **uncertainty as a first-class citizen** and operates under four calibrated outcome bands:

| Band | Meaning | Core Criteria |
| :--- | :--- | :--- |
| 🟣 **Declared AI** | Provenance flags AI origin | Embedded C2PA / JUMBF, IPTC XMP, or EXIF metadata explicitly states generative origin. |
| 🟢 **Likely Authentic** | No evidence of manipulation | All forensic and rule checks executed and found no anomalies. *(Never states "this is real")* |
| 🟡 **Uncertain** | Not enough to call it | Ambiguous signals, conflicting evidence, or degrading conditions (e.g., severe compression). |
| 🔴 **Likely Manipulated** | Multi-signal manipulation | Multiple independent forensic checks, heuristic rules, or vision signals agree on tampering. |

### The Core Architectural Tenet: "Zero Fabrication"
Every signal and finding shown to a user is backed by a deterministic or measured metric actually computed from the input bytes. If a library or check cannot run (e.g., missing decoder, undecodable format, blank frames), TrustLens **abstains** and logs a `LIMITATION` signal rather than producing synthetic or hallucinated numbers.

---

## 2. On Which Process and Pipeline Does It Work?

TrustLens can run under two distinct runtime processes:

```
                                  ┌───────────────────────────────┐
                                  │       Client Browser          │
                                  └──────────────┬────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │ (Option A: Primary)                             │ (Option B: Fallback)
                        ▼                                                 ▼
        ┌───────────────────────────────┐                 ┌───────────────────────────────┐
        │   FastAPI / Uvicorn Server    │                 │       Streamlit Server        │
        │        (`api.py`:8000)        │                 │        (`app.py`:8501)        │
        │  Serves React SPA + REST API  │                 │    Direct Interactive UI      │
        └──────────────┬────────────────┘                 └──────────────┬────────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                ▼
                                ┌───────────────────────────────┐
                                │   TrustLens Engine Ingestion  │
                                │   (`detectors.detect_modality`)│
                                └──────────────┬────────────────┘
                                                │
         ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
         ▼                  ▼                   ▼                   ▼                  ▼
    ┌──────────┐      ┌───────────┐       ┌───────────┐       ┌───────────┐     ┌─────────────┐
    │Text Lane │      │Image Lane │       │Audio Lane │       │Video Lane │     │Privacy Guard│
    │(Regex +  │      │(Forensics,│       │(FFmpeg PCM│       │(Frames +  │     │(Scratch file│
    │Stylometry│      │ELA, Ghosts│       │Spectrogram│       │Late Audio │     │ shredding & │
    │+ LLM)    │      │& C2PA)    │       │Forensics) │       │Fusion)    │     │ in-memory)  │
    └────┬─────┘      └─────┬─────┘       └─────┬─────┘       └─────┬─────┘     └──────┬──────┘
         │                  │                   │                   │                  │
         └──────────────────┴───────────────────┼───────────────────┴──────────────────┘
                                                ▼
                                ┌───────────────────────────────┐
                                │    Evidence Calibration       │
                                │  Noisy-OR Aggregation Engine  │
                                │    (`schemas.combine/decide`) │
                                └──────────────┬────────────────┘
                                                ▼
                                ┌───────────────────────────────┐
                                │   Explainer Engine            │
                                │ Template / Grounded Gemini LLM│
                                │    (`explainer.explain`)      │
                                └──────────────┬────────────────┘
                                                ▼
                                ┌───────────────────────────────┐
                                │ Final Structured JSON Result  │
                                └───────────────────────────────┘
```

### The 6-Step Processing Flow:
1. **Intake & Modality Routing (`detectors.detect_modality`)**:
   - The payload is examined based on MIME Content-Type first, then file extension.
   - File names are never inspected for semantic keywords (a file named `deepfake.mp4` is treated identically to `video.mp4`).
2. **Deterministic Forensic Inspection**:
   - Specific modality detectors execute in-memory analyses (or temporary atomic scratch files for video).
3. **External Model Inference (Optional)**:
   - When a `GEMINI_API_KEY` is present, TrustLens can execute grounded classification and **SelfCheckGPT** divergence testing across multiple samples.
   - If no key is configured or network is unavailable, it gracefully runs in forensic-only mode without crashing.
4. **Evidence Synthesis & Noisy-OR Fusion (`schemas.combine`)**:
   - Individual signal contributions $c_i \in [0, 0.9]$ accumulate probabilistically:
     $$\text{Remaining} = (1 - \text{Base}) \times \prod (1 - c_i)$$
     $$\text{Score} = 1 - \text{Remaining}$$
   - No single forensic signal can unilaterally jump to a 100% manipulated score.
5. **Calibrated Decision & Confidence Interval (`schemas.decide`)**:
   - Decides among `DECLARED_AI`, `LIKELY_AUTHENTIC`, `UNCERTAIN`, or `LIKELY_MANIPULATED`.
   - Incorporates hard out-of-distribution (`ood_flags`) and soft degradation flags (`soft_flags`, like JPEG compression).
6. **Strictly Grounded Explanation Generation (`explainer.py`)**:
   - Validates explanations against extracted signals. If the model introduces unmeasured terms (e.g. claims a "face swap" when no facial anomaly was measured), it is rejected and a deterministic template explanation is rendered.

---

## 3. Which Parts Are Working? (Feature & Lane Audit)

All core components are **100% operational** and validated against 52 test fixtures.

### A. Text Lane (`trustlens/detectors/text.py`)
* **Status:** **FULLY FUNCTIONAL**
* **Working Features:**
  - **11 Deterministic Scam/Phishing Rules:** Evaluates credential harvesting (OTP, PINs), payment demands (crypto, wire, gift cards), artificial urgency, account closure threats, shortened URLs, raw IP URLs, lookalike domains (homograph attacks), punycode (`xn--`), lottery/inheritance bait, authority impersonation, and chat redirection.
  - **Quoted Evidence Highlighting:** Directly quotes the exact character string in the text that triggered each rule.
  - **Stylometry & Burstiness Analysis (`stylometry.py`):** Calculates sentence length variance, lexical diversity (Type-Token Ratio / Simpson's D), and punctuation regularity to detect machine-generated assistant prose without relying on heavy transformers.
  - **SelfCheck Sampling (Manakul et al. 2023):** With Gemini LLM enabled, factual claims are verified across independent samples.

### B. Image Lane (`trustlens/detectors/image.py` & `forensics_image.py`)
* **Status:** **FULLY FUNCTIONAL**
* **Working Features:**
  - **Provenance & Metadata Inspection:** Extracts C2PA JUMBF boxes, IPTC DigitalSourceType / WebStatement, and EXIF camera parameters. Directly flags declared AI imagery (DALL-E, Midjourney, Firefly, Imagen).
  - **Noise Residual Analysis (Spatial PRNU-approximation):** Calculates Gaussian difference residual maps across $64\times64$ blocks. Identifies statistical outlier regions ($Z > 3.0$) indicative of spliced foreign elements.
  - **Error Level Analysis (ELA):** Measures resaved JPEG error variance across block matrices.
  - **JPEG Ghost Inversion (Farid 2009):** Steps through quality scales 50–98 to discover discrepancy in prior compression history.
  - **Haar Face Localization:** Employs OpenCV Haar cascades with a 1.3× boundary margin per FaceForensics++ specifications.
  - **Vision Model Inspection (`vision_llm.py`):** Optional multimodal artifact verification with SelfCheck consistency filtering.

### C. Audio Lane (`trustlens/detectors/audio.py` & `forensics_audio.py`)
* **Status:** **FULLY FUNCTIONAL**
* **Working Features:**
  - **Decoder Engine:** Pure in-memory streaming via `imageio-ffmpeg` to 16 kHz mono float PCM arrays.
  - **Spectral Cutoff Detection:** Identifies artificial low-pass brickwall filter ceilings (common in older neural vocoders at 4 kHz, 8 kHz, or 12 kHz).
  - **Digital Silence Check:** Flags mathematically exact zeros (`x == 0.0`) or unnatural step drops characteristic of synthetic speech boundaries.
  - **Autocorrelation Pitch Tracking:** Evaluates fundamental frequency ($F_0$) consistency and unnaturally static micro-prosody.
  - **Dynamic Noise Floor & Clipping:** Measures background noise levels relative to peak RMS.

### D. Video Lane (`trustlens/detectors/video.py`)
* **Status:** **FULLY FUNCTIONAL**
* **Working Features:**
  - **Temporal Frame Sampling:** Uniformly extracts 12 sample frames across duration.
  - **Distribution-Aware Reporting:** Reports the proportion of flagged frames (e.g., *"4 of 12 frames show anomalies"*) instead of a flattened average that dilutes localized tampering.
  - **Multi-modal Demuxing:** Simultaneously demuxes the underlying audio stream, runs the entire audio forensic lane, and late-fuses the results.
  - **Flat/Blank Frame Filtering:** Automatically discards near-black or uninformative transitional frames to prevent false positives.

### E. Privacy & Zero-Retention Subsystem (`trustlens/privacy.py`)
* **Status:** **FULLY FUNCTIONAL**
* **Working Features:**
  - **100% In-Memory Processing:** Text, images, and audio payloads never touch the disk.
  - **Secure Video Scratch File:** Because OpenCV `VideoCapture` requires a filesystem path, videos use a temporary scratch context manager that zeroes out bytes (`\0` shredding) and unlinks files in a `finally` block even if the detector crashes.
  - **Auditable Metrics:** Exposed via `GET /privacy` showing total created vs total removed scratch files.

### F. User Interfaces & API
* **Status:** **FULLY FUNCTIONAL**
* **Working Features:**
  - **Modern SPA (`web/`):** Built with React 18 and Framer Motion, vendored locally in `web/vendor` (zero external CDN or internet dependency required, no Node.js compilation required).
  - **REST API (`api.py`):** Fully documented FastAPI endpoints at `/docs` (`/analyze`, `/analyze/text`, `/health`, `/privacy`).
  - **Streamlit App (`app.py`):** Complete fallback UI with visual meters, evidence cards, and raw technical drawers.

---

## 4. Summary Table of Capabilities & Limitations

| Capability | Status | Implementation Details / Limitations |
| :--- | :---: | :--- |
| **Scam / Phishing Detection** | ✅ Works | 11 deterministic pattern rules + LLM reasoning |
| **AI Text Stylometry** | ✅ Works | Burstiness, entropy, sentence variance (no heavy neural models needed) |
| **Image Splicing / Editing** | ✅ Works | Noise residual Z-score clustering + ELA + JPEG ghosts |
| **Metadata Provenance** | ✅ Works | C2PA, IPTC, EXIF parsed directly |
| **Voice Processing Forensics**| ✅ Works | Spectral ceiling, digital silence, pitch stability |
| **Video Deepfake Detection** | ✅ Works | Frame sampling + audio demuxing late fusion |
| **Zero-Disk Retention** | ✅ Works | In-memory + atomic file shredder |
| **Subtle Face Identity Swap** | ⚠️ Limited | Deliberate limitation: no heavy Torch face-swap CNN bundled (as explained in AGENTS.md) |
| **Full AI Image Generation** | ⚠️ Conditional | Requires `GEMINI_API_KEY` for synthetic diffusion recognition; without key, detects edits & provenance |
| **Mic-recorded Voice Clones** | ⚠️ Limited | High-quality voice clones played over an analog microphone pass audio processing checks |

---

## 5. How to Run the Project

### Running the Test Suite
```bash
py -m unittest discover tests -v
```
*(Executes all 52 unit tests offline using `TRUSTLENS_NO_MODEL=1`)*

### Starting the Primary Web App & API
```bash
py -m uvicorn api:app --reload
```
- **Web Interface:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Starting the Streamlit Fallback App
```bash
py -m streamlit run app.py
```
- **Streamlit UI:** [http://localhost:8501](http://localhost:8501)
