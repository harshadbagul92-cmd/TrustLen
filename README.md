# TrustLens

A classifier and explainer for **scam text, manipulated images, cloned voices and deepfake video** — which tells you *why* it flagged something, in plain words, and says "I don't know" when the evidence doesn't support an answer.

---

## Why it refuses to say "fake"

One finding shaped this whole project. **Deepfake-Eval-2024** took nine state-of-the-art open-source detectors and ran them on deepfakes collected from real social media instead of on the benchmarks they were built for:

| Model | On its own benchmark | In the wild |
|---|---|---|
| AASIST (audio) | 1.00 AUC | **0.43** |
| GenConViT (video) | 0.96 | **0.63** |
| NPR (image) | 0.98 | **0.53** |

0.50 is a coin flip. The best *commercial* detectors the same team tested reached 0.78 / 0.89 / 0.82 — **none hit 90%**.

So TrustLens never returns a binary verdict. It uses four bands:

| Band | Meaning |
|---|---|
| 🟣 **Declared AI** | The file's own metadata says it was AI-generated |
| 🟢 **No evidence of manipulation** | Checks ran and found nothing. Never "this is real" |
| 🟡 **Not enough to call it** | Ambiguous, *or* conditions that make the checks unreliable |
| 🔴 **Likely manipulated** | Multiple independent checks agree |

**The core rule:** every finding shown to a user corresponds to a measurement that was actually taken. When a check can't run, the app says so. It never fills the gap with a plausible-sounding number.

---

## Install

```bash
pip install -r requirements.txt
```

No torch, no transformers. Under 200 MB, a few minutes.

Then add your API key (optional but recommended). Copy `.env.example` to `.env` and paste the key in:

```
GEMINI_API_KEY=your-key-here
```

Get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). `.env` is gitignored.

Without a key everything still runs on rules and forensics — the app states on every result that it cannot detect fully AI-generated content in that mode.

**Two things worth knowing about the model:**

- The default is `gemini-flash-lite-latest`. It is an *alias*, so it does not go stale — a pinned `gemini-2.0-flash` was retired by Google and every call silently began returning 404. And the lite model has far more free-tier quota than the premium one, which two or three analyses can exhaust.
- If a model is rate-limited, TrustLens falls through `MODEL_FALLBACKS` to the next one rather than losing the check mid-demo. A rate limit also cools down and retries, instead of disabling the model for the whole session.

To run with no network at all, set `TRUSTLENS_NO_MODEL=1`.

## Run

One command serves both the interface and the API:

```bash
uvicorn api:app --reload
```

- Interface — **http://127.0.0.1:8000**
- API docs — http://127.0.0.1:8000/docs

There is also a plainer Streamlit version kept as a fallback, in case anything
goes wrong with the main one on the day:

```bash
streamlit run app.py
```

### About the interface

React 18 + Framer Motion, with **no Node.js and no build step**. The libraries
are vendored under `web/vendor/` (274 KB total) rather than loaded from a CDN,
so the interface still works on a dead network — which is exactly when a demo
needs it to. Edit `web/app.js` or `web/styles.css` and refresh; there is
nothing to compile.

```
web/
  index.html    shell and script tags
  styles.css    design tokens, both themes, every animation
  app.js        the whole React app (htm templates, no JSX compiler)
  vendor/       React, ReactDOM, Framer Motion, htm
```

Four sidebar tabs: **Analyse**, **How it works**, **Privacy**, **System**.
Dark by default with a light toggle; the choice is remembered.

One rule the interface follows: **blue is the brand and never means a verdict.**
Green, amber, red and violet carry meaning only, so a result reads correctly
from across a room.

## Test

```bash
python -m unittest discover tests -v
```

32 tests. All fixtures are generated at run time, so it passes on a fresh clone with no sample data committed. The suite pins `TRUSTLENS_NO_MODEL=1`, so it never calls the network or spends your quota.

---

## What each lane actually does

### Text — scam, phishing, misinformation
- **11 deterministic rules** with the matched text quoted back, so a person can judge for themselves. Works offline.
- **Model classification** reading the message *and* the rule hits, so it reasons over evidence.
- **SelfCheck** (Manakul et al. 2023): the central factual claim is put to the model N times; answers that contradict each other mean the claim isn't well supported.

> Deliberately not implemented: scoring on spelling or grammar. It correlates with some scams, but it penalises anyone writing in a second language, and a fluent scam sails past it.

### Image — synthetic and tampered
- **Provenance first** — C2PA/JUMBF, IPTC XMP, EXIF, read from real file structure. Costs milliseconds and has near-zero false positives.
- **Noise residual** — the primary localiser. Finds regions carrying a different sensor-noise texture.
- **ELA** and **JPEG ghost** (Farid 2009) — corroboration only. Measured on real photos, ELA fired on hair detail and missed an actual splice, so it never accuses on its own.
- **Faces** cropped at **1.3×** per FaceForensics++, where that crop was worth ~17 accuracy points over whole-image input.
- **Vision model** with SelfCheck filtering — approximating SIDA (Huang et al. 2024).

### Audio — how the recording was processed
Honest scope: research-grade anti-spoofing needs torch, and even AASIST falls to 0.43 AUC in the wild. This lane measures **processing**, not synthesis:
- Hard spectral ceiling, digital-exact silence, pitch steadiness, noise floor, clipping.
- Every audio result carries a standing limitation: *a high-quality clone recorded through a real microphone would pass these checks.*

### Video — frames plus the voice track
- Samples 12 frames, runs the cheap forensic subset on each.
- **Reports the distribution, not the average** — "5 of 12 frames" — because published models assume the whole video is fake and lose ~31% accuracy on selectively manipulated video.
- Demuxes the audio and runs the audio lane, then late-fuses.

---

## Privacy

**Nothing you submit is stored.** Uploads are analysed in memory and discarded when the result renders.

The one unavoidable exception is video: OpenCV can only read from a path. That goes through `trustlens/privacy.py`, which writes to a scratch file, overwrites it, and deletes it **in a `finally` block** — so it goes even if the analysis crashes.

This is auditable: grep for `scratch_file` and you have found every write of user content. `GET /privacy` reports the live counters, and there's a test asserting nothing survives.

---

## Layout

```
trustlens/
  config.py              every tunable threshold, with what it was measured against
  schemas.py             the Evidence contract every lane returns
  privacy.py             scratch-file lifecycle; the only place user content touches disk
  llm.py                 Gemini wrapper + SelfCheck sampling
  explainer.py           plain-English output, grounded and scope-checked
  detectors/
    text.py              rules + classifier + SelfCheck
    image.py             provenance + forensics + vision model
    audio.py             acoustic measurement
    video.py             frames + audio, late fusion
    forensics_image.py   noise residual, ELA, JPEG ghost, provenance, faces
    forensics_audio.py   spectrum, pitch, silence, noise floor
    media.py             ffmpeg decode, frame sampling
    vision_llm.py        SIDA-style assessment with consistency filtering
app.py                   Streamlit interface
api.py                   FastAPI service
tests/                   32 tests, self-generating fixtures, offline
```

**To change how strict it is**, edit `trustlens/config.py` — every threshold is there with a note on what it was measured against.

**To add a lane**, subclass `Detector`, return an `Evidence`, register it in `detectors/__init__.py`. The UI and explainer need no changes.

---

## What this cannot do

Stated here because it's stated in the app too:

- **No trained face-swap detector.** A subtle identity swap will be missed.
- **Without an API key**, it finds *edits*, not fully generated images.
- **Audio checks describe processing, not synthesis.** A good clone through a real mic passes them.
- **Thresholds are tuned on a small sample.** They are a starting point, not a validated benchmark.

Accuracy figures quoted above belong to the cited papers, not to this build.

---

## Sources

FaceForensics++ (Rössler et al., 2019) · SIDA (Huang et al., 2024) · AASIST (Jung et al., 2021) · Deepfake-Eval-2024 (Chandra et al.) · SelfCheckGPT (Manakul et al., 2023) · SynthID-Image (Google DeepMind, 2025) · JPEG Ghosts (Farid, 2009)
