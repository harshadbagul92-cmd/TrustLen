"""
Every tunable number in TrustLens lives here.

Tuning thresholds is the main thing you will want to change, so they are in
one file rather than scattered through the detectors. Each one records what
it was measured against, so you can tell whether a change is safe.
"""
import os

# ---------------------------------------------------------------------------
# Verdict bands
# ---------------------------------------------------------------------------
# Deliberately NOT a binary real/fake split. Deepfake-Eval-2024 measured
# published detectors dropping from 0.87-1.00 AUC on their own test sets to
# 0.43-0.63 on real-world media, and no commercial detector tested reached
# 90% accuracy. A confident binary claim is not supportable, so there is an
# explicit "we cannot tell" band in the middle.
AUTHENTIC_BELOW = 0.35
MANIPULATED_ABOVE = 0.65

BASE_SCORE = 0.10          # score when every check ran and found nothing
ABSTAIN_SCORE = 0.50       # score used when we genuinely cannot judge
BASE_CONFIDENCE_MARGIN = 0.05

# ---------------------------------------------------------------------------
# Image forensics
# ---------------------------------------------------------------------------
IMG_Z_THRESHOLD = 3.5          # robust z above which a block is anomalous
IMG_NOISE_BLOCK = 32
IMG_ELA_BLOCK = 16
IMG_NOISE_MIN_BLOCKS = 6       # a finding must be an area, not one hot block
IMG_ELA_MIN_BLOCKS = 8
IMG_MIN_COVERAGE_PCT = 0.5     # genuine photos gave 0.2% specks; a real
                               # splice covered 1.2% of the frame
IMG_GHOST_DEV_STEPS = 1.5      # quality-sweep steps from the dominant fit
IMG_MIN_GRADIENT = 5.0         # below this the image is too flat to analyse
IMG_MIN_PIXEL_STD = 12.0
IMG_FACE_CROP_SCALE = 1.3      # FaceForensics++ found this crop worth ~17
                               # accuracy points over whole-image input
IMG_FACE_MIN_SIZE_FRAC = 0.08  # ignore faces smaller than this fraction of
                               # the frame - they are usually posters or
                               # reflections, not the subject

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16000
AUDIO_WINDOW_SECONDS = 4.0     # analysis window
AUDIO_MAX_SECONDS = 120        # cap work on long uploads
AUDIO_MIN_SECONDS = 0.8        # below this there is nothing to measure

# A hard spectral ceiling is the single most reliable honest cue available
# without a trained model: many synthesis and re-encoding pipelines discard
# everything above a fixed frequency, which natural wideband speech does not.
AUDIO_BANDWIDTH_SUSPICIOUS_HZ = 7800
AUDIO_PITCH_CV_FLAT = 0.045    # coefficient of variation of F0 below which a
                               # voice is unusually steady
AUDIO_CLIPPING_PCT = 1.0

# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
VIDEO_SAMPLE_FRAMES = 12
VIDEO_MAX_SECONDS = 300
VIDEO_FRAME_FLAG_RATIO = 0.34  # share of sampled frames that must be flagged
                               # before the video itself is called suspicious

# ---------------------------------------------------------------------------
# Vision / language model
# ---------------------------------------------------------------------------
# SelfCheckGPT (Manakul et al. 2023): sample N times, keep only what the model
# repeats. Agreement across samples IS the confidence - a model's own stated
# confidence is not trustworthy.
# Two samples, not three. Each sample is an API call, and the free Gemini tier
# runs out fast: one image at 3 samples plus one text analysis at 5 calls was
# enough to hit a 429. With 2 samples a "consistent" observation must appear in
# BOTH looks, which is stricter as well as cheaper.
VLM_SAMPLES = 2

# Two lessons are baked into these defaults.
#
# 1. Use a moving alias, not a pinned version. A pinned "gemini-2.0-flash" was
#    retired by Google and every call began returning 404 - the app still ran,
#    but silently lost its vision and language checks.
# 2. Default to the LITE alias. "gemini-flash-latest" points at the current
#    premium model, whose free-tier quota is tiny; a couple of analyses
#    exhausted it and every later call returned 429. The lite model answers
#    text and vision equally well for this task and has far more headroom.
VLM_MODEL = os.environ.get("TRUSTLENS_VLM_MODEL", "gemini-flash-lite-latest")
TEXT_MODEL = os.environ.get("TRUSTLENS_TEXT_MODEL", "gemini-flash-lite-latest")

# Tried in order when the model above is rate-limited, so one exhausted quota
# does not take the demo down mid-presentation.
MODEL_FALLBACKS = ["gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-flash-latest"]

API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")

# ---------------------------------------------------------------------------
# Limits and privacy
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = 200
DEBUG = os.environ.get("TRUSTLENS_DEBUG", "").lower() in ("1", "true", "yes")

# Set TRUSTLENS_NO_MODEL=1 to force the offline path even when a key is set.
# The test suite uses it so tests stay fast, deterministic and free; it is also
# the switch to flip if you want to demo without touching the network.
NO_MODEL = os.environ.get("TRUSTLENS_NO_MODEL", "").lower() in ("1", "true", "yes")


def _load_dotenv() -> None:
    """
    Read a local .env file into the environment, once, at import.

    Hand-rolled rather than pulling in python-dotenv: it is fifteen lines, and
    one fewer dependency matters more than the extra features. Real environment
    variables always win, so a shell export overrides the file.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass          # a malformed .env must never stop the app starting


_load_dotenv()


def api_key() -> str | None:
    """The Gemini key, or None. Never logged, never echoed to the UI."""
    for var in API_KEY_ENV_VARS:
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip()
    return None


def has_api_key() -> bool:
    return api_key() is not None
