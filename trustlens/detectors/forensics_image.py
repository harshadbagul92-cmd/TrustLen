"""
Image forensics primitives.

Measured against real photographs and a splice with known ground truth:
  - noise_residual_zmap  hit the splice, silent on all genuine photos  -> leads
  - ela_zmap             fired on hair detail, MISSED the splice       -> second opinion only
  - jpeg_ghost_map       zero false positives once flat blocks masked  -> corroboration only

That ordering is why image.py treats the noise map as the primary localiser
and lets the other two confirm but never accuse on their own.

EVERY function here returns a MEASUREMENT taken from the actual pixels or the
actual file bytes. Nothing in this module invents, guesses, or narrates.
If a measurement cannot be taken, the function returns None and the caller
must treat that as "no evidence", never as "no manipulation".

No torch, no transformers. numpy + Pillow + OpenCV only.
"""

import io
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# JPEG quantization tables
# ---------------------------------------------------------------------------

# Annex K standard luminance quantization table, natural (row-major) order.
_STD_LUMA_NATURAL = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

# Zigzag scan order: ZIGZAG[k] is the natural-order index of zigzag position k.
_ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]

# PIL reports quantization tables in zigzag order, so reorder the standard
# table to match before pairing coefficients.
_STD_LUMA_ZIGZAG = [_STD_LUMA_NATURAL[_ZIGZAG[k]] for k in range(64)]


def jpeg_quality_from_qtables(pil_img: Image.Image) -> Optional[int]:
    """
    Invert the IJG quality->table relationship to recover the encoder quality
    setting actually used on this file.

    IJG forward:  scale = 5000/Q  (Q < 50)  or  200 - 2Q  (Q >= 50)
                  table[i] = floor((std[i] * scale + 50) / 100)

    We invert per coefficient and take the median, which is robust to the
    clamping that happens on very large/small coefficients.

    Returns None when the image carries no quantization tables (e.g. PNG).
    """
    tables = getattr(pil_img, "quantization", None)
    if not tables:
        return None
    luma = tables.get(0)
    if luma is None or len(luma) < 64:
        return None

    qualities: List[float] = []
    for k in range(64):
        q = float(luma[k])
        std = float(_STD_LUMA_ZIGZAG[k])
        if q <= 0 or std <= 0:
            continue
        scale = (q * 100.0 - 50.0) / std
        if scale <= 0:
            continue
        if scale > 100.0:
            quality = 5000.0 / scale
        else:
            quality = (200.0 - scale) / 2.0
        qualities.append(max(1.0, min(100.0, quality)))

    if not qualities:
        return None
    return int(round(float(np.median(qualities))))


# ---------------------------------------------------------------------------
# Provenance: C2PA / JUMBF / XMP / EXIF, read from real file structure
# ---------------------------------------------------------------------------

def _iter_jpeg_segments(data: bytes):
    """Yield (marker, payload) for each JPEG APPn/COM segment."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return
    i = 2
    n = len(data)
    while i < n - 3:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:      # start of scan - entropy data follows, stop
            return
        if i + 4 > n:
            return
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        payload = data[i + 4: i + 2 + seg_len]
        yield marker, payload
        i += 2 + seg_len


def read_provenance(image_bytes: Optional[bytes], pil_img: Optional[Image.Image]) -> Dict[str, Any]:
    """
    Look for an explicit machine-readable declaration of AI origin.

    Only three things count, and each is checked against real file structure
    rather than a substring search over the whole binary (which false-positives
    on compressed pixel data):
      1. A JUMBF/C2PA box in an APP11 segment.
      2. An XMP packet in APP1 carrying IPTC digitalSourceType.
      3. An EXIF Software/Artist tag naming a known generator.

    The filename is NEVER consulted. Renaming a file must not change the verdict.
    """
    out: Dict[str, Any] = {
        "declared_ai": False,
        "method": None,
        "detail": None,
        "exif_tag_count": 0,
        "has_c2pa_container": False,
    }
    if not image_bytes:
        return out

    # --- 1. C2PA lives in a JUMBF box inside APP11 -------------------------
    for marker, payload in _iter_jpeg_segments(image_bytes):
        if marker == 0xEB:                      # APP11
            if b"jumb" in payload[:64] or b"c2pa" in payload[:256].lower():
                out["has_c2pa_container"] = True
                low = payload.lower()
                # c2pa.ai.generativeTraining / trainedAlgorithmicMedia assertions
                if b"trainedalgorithmicmedia" in low or b"c2pa.actions" in low:
                    out["declared_ai"] = True
                    out["method"] = "c2pa_jumbf"
                    out["detail"] = "C2PA Content Credentials assert AI generation"
                    return out

        # --- 2. XMP packet in APP1 ----------------------------------------
        if marker == 0xE1 and payload[:29] == b"http://ns.adobe.com/xap/1.0/\x00":
            xmp = payload[29:].decode("utf-8", errors="ignore")
            low = xmp.lower()
            if "digitalsourcetype" in low and (
                "trainedalgorithmicmedia" in low
                or "compositewithtrainedalgorithmicmedia" in low
            ):
                out["declared_ai"] = True
                out["method"] = "iptc_xmp_digitalSourceType"
                out["detail"] = "IPTC digitalSourceType = trainedAlgorithmicMedia"
                return out

    # --- 3. EXIF software / artist ----------------------------------------
    generators = [
        "midjourney", "dall-e", "dalle", "stable diffusion", "stablediffusion",
        "firefly", "flux", "novelai", "comfyui", "fooocus", "imagen",
        "leonardo.ai", "ideogram", "gemini", "grok", "openai",
    ]
    if pil_img is not None:
        try:
            exif = pil_img.getexif()
            out["exif_tag_count"] = len(exif)
            for tag_id in (305, 315, 270):          # Software, Artist, ImageDescription
                val = exif.get(tag_id)
                if not val:
                    continue
                text = str(val).lower()
                for g in generators:
                    if g in text:
                        out["declared_ai"] = True
                        out["method"] = "exif_software"
                        out["detail"] = f"EXIF tag names '{str(val)[:60]}'"
                        return out
        except Exception:
            pass

    return out


def exif_completeness(pil_img: Optional[Image.Image]) -> Dict[str, Any]:
    """
    Count EXIF tags and note whether camera-characteristic tags are present.

    IMPORTANT CAVEAT for the caller: messaging apps strip EXIF from genuine
    photographs. Absent EXIF is therefore a very weak hint and must never
    carry meaningful weight on its own.
    """
    result = {"tag_count": 0, "has_camera_tags": False, "make_model": None}
    if pil_img is None:
        return result
    try:
        exif = pil_img.getexif()
        result["tag_count"] = len(exif)
        make, model = exif.get(271), exif.get(272)
        if make or model:
            result["has_camera_tags"] = True
            result["make_model"] = f"{make or ''} {model or ''}".strip()
    except Exception:
        pass
    return result


# Exact output sizes commonly emitted by diffusion models.
_GENERATOR_SIZES = {
    (512, 512), (768, 768), (1024, 1024), (1536, 1536), (2048, 2048),
    (1024, 1536), (1536, 1024), (1344, 768), (768, 1344),
    (1152, 896), (896, 1152), (1216, 832), (832, 1216),
}


def generator_dimension_hint(size: Tuple[int, int]) -> Optional[str]:
    """Flag exact canvas sizes typical of diffusion models. Weak hint only."""
    if tuple(size) in _GENERATOR_SIZES:
        return f"{size[0]}x{size[1]} is an exact size commonly emitted by image generators"
    return None


# ---------------------------------------------------------------------------
# Localized forensics: ELA and noise residual, both texture-normalized
# ---------------------------------------------------------------------------

def _block_reduce(arr: np.ndarray, block: int) -> np.ndarray:
    """Mean-pool a 2-D array into non-overlapping blocks."""
    h, w = arr.shape
    bh, bw = h // block, w // block
    if bh < 2 or bw < 2:
        return np.zeros((0, 0), dtype=np.float32)
    trimmed = arr[:bh * block, :bw * block]
    return trimmed.reshape(bh, block, bw, block).mean(axis=(1, 3))


def _robust_z(arr: np.ndarray) -> np.ndarray:
    """Median/MAD z-score. Robust to the few genuinely odd blocks we're hunting."""
    if arr.size == 0:
        return arr
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = float(arr.std()) or 1.0
    return (arr - med) / scale


def _texture_energy(gray: np.ndarray, block: int) -> np.ndarray:
    """Per-block gradient magnitude. Used to normalise texture-driven response."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return _block_reduce(mag, block)


def ela_zmap(pil_img: Image.Image, quality: int = 90, block: int = 16
             ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Error Level Analysis, block-pooled and normalised by local texture.

    Raw ELA responds strongly to edges, so a plain ELA map just highlights
    every edge in the picture. Dividing by local gradient energy removes most
    of that, leaving blocks whose re-compression error is out of proportion
    to their own detail - the actual signature of a region with a different
    compression history.

    Returns (z-score map per block, stats) or (None, stats) on failure.
    """
    stats: Dict[str, Any] = {"block": block, "resave_quality": quality}
    try:
        buf = io.BytesIO()
        pil_img.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")

        a = np.asarray(pil_img, dtype=np.float32)
        b = np.asarray(resaved, dtype=np.float32)
        if a.shape != b.shape:
            return None, stats
        diff = np.abs(a - b).mean(axis=2)

        gray = np.asarray(pil_img.convert("L"), dtype=np.float32)
        ela_blocks = _block_reduce(diff, block)
        tex_blocks = _texture_energy(gray, block)
        if ela_blocks.size == 0 or tex_blocks.shape != ela_blocks.shape:
            return None, stats

        ratio = ela_blocks / (tex_blocks + 1.0)
        z = _robust_z(ratio)

        stats["mean_abs_diff"] = round(float(diff.mean()), 4)
        stats["blocks"] = f"{z.shape[0]}x{z.shape[1]}"
        stats["max_z"] = round(float(z.max()), 2) if z.size else None
        return z, stats
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return None, stats


def jpeg_ghost_map(pil_img: Image.Image, block: int = 16,
                   q_lo: int = 55, q_hi: int = 96, q_step: int = 5
                   ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    JPEG ghost analysis (Farid, "Exposing Digital Forgeries from JPEG Ghosts",
    IEEE TIFS 2009).

    Re-save the image across a sweep of qualities. For each block, the quality
    whose re-save error is LOWEST is roughly the quality that block was last
    compressed at. A region imported from another photograph carries its own
    compression history, so its best-fit quality sits at a different point in
    the sweep than the rest of the frame - a "ghost".

    Returns (deviation map in sweep steps, stats). The deviation is how far a
    block's best-fit quality is from the image's dominant best-fit quality,
    which is directly interpretable rather than an abstract z-score.
    """
    stats: Dict[str, Any] = {"block": block, "q_range": [q_lo, q_hi, q_step]}
    try:
        base = np.asarray(pil_img.convert("RGB"), dtype=np.float32)
        qualities = list(range(q_lo, q_hi + 1, q_step))
        layers = []
        for q in qualities:
            buf = io.BytesIO()
            pil_img.save(buf, "JPEG", quality=int(q))
            buf.seek(0)
            resaved = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
            if resaved.shape != base.shape:
                return None, stats
            err = ((base - resaved) ** 2).mean(axis=2)
            pooled = _block_reduce(err, block)
            if pooled.size == 0:
                return None, stats
            layers.append(pooled)

        cube = np.stack(layers, axis=0)             # (n_q, bh, bw)
        best_q_idx = np.argmin(cube, axis=0).astype(np.int32)

        # Flat blocks re-save almost losslessly at EVERY quality, so their
        # argmin is arbitrary noise. Measured on samples/, leaving them in
        # produced large false regions over dark sky and shadow; masking them
        # out removed every false positive. Only textured blocks may vote.
        gray = np.asarray(pil_img.convert("L"), dtype=np.float32)
        tex = _texture_energy(gray, block)
        if tex.shape != best_q_idx.shape:
            return None, stats
        mask = tex >= 8.0
        stats["textured_blocks"] = int(mask.sum())
        if mask.sum() < 10:
            stats["skipped"] = "too few textured blocks"
            return None, stats

        counts = np.bincount(best_q_idx[mask].ravel(), minlength=len(qualities))
        dominant = int(np.argmax(counts))
        deviation = np.abs(best_q_idx - dominant).astype(np.float32)
        deviation[~mask] = 0.0

        stats["qualities"] = qualities
        stats["dominant_quality"] = qualities[dominant]
        stats["dominant_share_pct"] = round(100.0 * counts[dominant] / int(mask.sum()), 1)
        stats["max_deviation_steps"] = int(deviation.max())
        stats["blocks"] = f"{deviation.shape[0]}x{deviation.shape[1]}"
        return deviation, stats
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return None, stats


def noise_residual_zmap(cv_img: np.ndarray, block: int = 32
                        ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Block-wise sensor-noise residual, normalised by local texture.

    Content pasted in from a different photograph carries that photograph's
    noise floor. After normalising for texture, blocks whose residual energy
    sits far from the image median are candidates for spliced content.
    """
    stats: Dict[str, Any] = {"block": block}
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.medianBlur(gray, 3).astype(np.float32)
        residual = np.abs(gray.astype(np.float32) - denoised)

        res_blocks = _block_reduce(residual, block)
        tex_blocks = _texture_energy(gray.astype(np.float32), block)
        if res_blocks.size == 0 or tex_blocks.shape != res_blocks.shape:
            return None, stats

        ratio = res_blocks / (tex_blocks + 1.0)
        z = _robust_z(ratio)

        stats["mean_residual"] = round(float(residual.mean()), 4)
        stats["blocks"] = f"{z.shape[0]}x{z.shape[1]}"
        stats["max_z"] = round(float(z.max()), 2) if z.size else None
        return z, stats
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return None, stats


def flatness(gray: np.ndarray) -> Dict[str, float]:
    """
    How much real detail does this image contain?

    Both forensic maps divide by local texture, so an image that is almost
    uniformly flat (a near-black frame, a blank canvas) produces meaningless
    ratios and wild z-scores. The caller must check this before trusting any
    region, or it will confidently "localise manipulation" in empty darkness.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "mean_gradient": float(np.sqrt(gx * gx + gy * gy).mean()),
        "pixel_std": float(gray.std()),
    }


def cluster_outliers(zmap: np.ndarray, block: int, img_w: int, img_h: int,
                     z_thresh: float = 3.5, min_blocks: int = 4,
                     min_coverage_pct: float = 0.0
                     ) -> List[Dict[str, Any]]:
    """
    Turn a block z-map into pixel-space regions.

    A finding must be both STRONG (peak z above threshold) and BIG ENOUGH
    (min_blocks, min_coverage_pct). Measured on samples/, genuine photographs
    produce occasional 1-2 block spikes on fine detail such as hair, while a
    real pasted region covered ~1.2% of the frame across ~30 blocks. Requiring
    size is what separates the two.
    """
    if zmap is None or zmap.size == 0:
        return []
    mask = (zmap > z_thresh).astype(np.uint8)
    if mask.sum() == 0:
        return []

    count, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions: List[Dict[str, Any]] = []
    for lbl in range(1, count):
        x, y, w, h, area = stats_cc[lbl]
        if area < min_blocks:
            continue
        if 100.0 * area / zmap.size < min_coverage_pct:
            continue
        px = (int(x * block), int(y * block), int((x + w) * block), int((y + h) * block))
        peak = float(zmap[labels == lbl].max())
        regions.append({
            "bbox": px,
            "area_blocks": int(area),
            "peak_z": round(peak, 2),
            "coverage_pct": round(100.0 * area / zmap.size, 2),
            "region": coarse_region_name((px[0] + px[2]) // 2, (px[1] + px[3]) // 2, img_w, img_h),
        })
    regions.sort(key=lambda r: r["peak_z"], reverse=True)
    return regions


def coarse_region_name(x: int, y: int, width: int, height: int) -> str:
    """Human-readable location, so the explainer can say where without coordinates."""
    nx = x / max(1, width)
    ny = y / max(1, height)
    vert = "upper" if ny < 0.35 else ("lower" if ny > 0.65 else "middle")
    horiz = "left" if nx < 0.35 else ("right" if nx > 0.65 else "centre")
    if vert == "middle" and horiz == "centre":
        return "the centre of the image"
    return f"the {vert}-{horiz} area"


# ---------------------------------------------------------------------------
# Faces
# ---------------------------------------------------------------------------

def detect_faces(cv_img: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], Optional[str]]:
    """
    Haar cascade face detection. Returns (boxes, error).

    Never raises: a detector crash must not take the whole lane down, but the
    caller does need to know it failed so it can flag the result.
    """
    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return [], "cascade_file_missing"
        # A fixed 40px floor picks up faces printed on posters and reflections
        # in the background. Scaling the floor to the frame keeps detections to
        # faces that are actually subjects of the photo. Trade-off: genuinely
        # small faces in a wide group shot will be missed.
        floor = max(40, int(min(cv_img.shape[:2]) * 0.08))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                         minSize=(floor, floor))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces], None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def crop_face_13x(cv_img: np.ndarray, box: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop the face enlarged by 1.3x, per FaceForensics++ (Rossler et al. 2019),
    where this crop outperformed whole-image input by a wide margin.
    """
    h_img, w_img = cv_img.shape[:2]
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * 1.3, h * 1.3
    x1 = max(0, int(cx - nw / 2)); y1 = max(0, int(cy - nh / 2))
    x2 = min(w_img, int(cx + nw / 2)); y2 = min(h_img, int(cy + nh / 2))
    crop = cv_img[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
