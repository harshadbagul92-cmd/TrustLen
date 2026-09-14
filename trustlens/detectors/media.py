"""
Decoding, via the ffmpeg binary that ships with imageio-ffmpeg.

No system ffmpeg install is needed - `imageio_ffmpeg.get_ffmpeg_exe()` returns
a bundled executable. Everything here is best-effort: if decoding fails the
caller gets None and reports an honest abstention.

Audio is decoded straight to raw mono float samples on stdout, so nothing
intermediate is ever written to disk. Video frames go through
privacy.scratch_file because OpenCV needs a real path.
"""
import subprocess
from typing import List, Optional, Tuple

import numpy as np

from .. import config, privacy

_FFMPEG: Optional[str] = None
_FFMPEG_ERROR: Optional[str] = None


def ffmpeg_path() -> Optional[str]:
    global _FFMPEG, _FFMPEG_ERROR
    if _FFMPEG or _FFMPEG_ERROR:
        return _FFMPEG
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        _FFMPEG_ERROR = type(exc).__name__
    return _FFMPEG


def ffmpeg_available() -> bool:
    return ffmpeg_path() is not None


def _run(args: List[str], stdin_data: Optional[bytes] = None,
         timeout: int = 120) -> Tuple[int, bytes, bytes]:
    proc = subprocess.Popen(
        args, stdin=subprocess.PIPE if stdin_data is not None else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        out, err = proc.communicate(input=stdin_data, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return -1, b"", b"timeout"
    return proc.returncode, out, err


def decode_audio(data: bytes, sample_rate: int = None,
                 max_seconds: int = None) -> Optional[np.ndarray]:
    """
    Any audio or video container -> mono float32 samples in [-1, 1].

    Piped in and out through memory; nothing touches the disk.
    """
    exe = ffmpeg_path()
    if not exe or not data:
        return None
    sample_rate = sample_rate or config.AUDIO_SAMPLE_RATE
    max_seconds = max_seconds or config.AUDIO_MAX_SECONDS

    code, out, _ = _run([
        exe, "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-t", str(max_seconds),
        "-vn",                      # ignore any video stream
        "-ac", "1",                 # mono
        "-ar", str(sample_rate),
        "-f", "f32le",              # raw float32 little-endian
        "pipe:1",
    ], stdin_data=data)

    if code != 0 or not out:
        return None
    samples = np.frombuffer(out, dtype=np.float32)
    if samples.size == 0:
        return None
    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)


def probe(data: bytes, suffix: str = "") -> dict:
    """Container facts: duration, stream count, resolution, fps."""
    exe = ffmpeg_path()
    info: dict = {}
    if not exe:
        return info
    # ffmpeg writes stream info to stderr when given no output file.
    code, _, err = _run([exe, "-hide_banner", "-i", "pipe:0"], stdin_data=data, timeout=60)
    text = err.decode("utf-8", errors="ignore")
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
    if m:
        h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        info["duration_s"] = round(h * 3600 + mnt * 60 + s, 2)
    m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", text)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
    if m:
        info["fps"] = float(m.group(1))
    info["has_audio_stream"] = "Audio:" in text
    info["has_video_stream"] = "Video:" in text
    m = re.search(r"encoder\s*:\s*(.+)", text)
    if m:
        info["encoder"] = m.group(1).strip()[:120]
    return info


def sample_frames(data: bytes, suffix: str, count: int = None
                  ) -> Tuple[List[np.ndarray], dict]:
    """
    Evenly spaced frames from a video, as BGR arrays.

    OpenCV cannot read a video from memory, so this is the one place user
    content reaches the filesystem - through privacy.scratch_file, which
    deletes it in a finally block.
    """
    import cv2
    count = count or config.VIDEO_SAMPLE_FRAMES
    frames: List[np.ndarray] = []
    meta: dict = {}

    with privacy.scratch_file(data, suffix or ".mp4") as path:
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return [], {"error": "opencv_could_not_open"}
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            meta = {"total_frames": total, "fps": round(fps, 2),
                    "resolution": f"{width}x{height}",
                    "duration_s": round(total / fps, 2) if fps > 0 else None}

            if total > 0:
                indices = np.linspace(0, max(0, total - 1), num=min(count, total)).astype(int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        frames.append(frame)
                        meta.setdefault("frame_indices", []).append(int(idx))
            else:
                # Streams without a frame count: read sequentially instead.
                read = 0
                while read < count * 10 and len(frames) < count:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if read % 10 == 0:
                        frames.append(frame)
                    read += 1
                meta["total_frames"] = read
        finally:
            cap.release()

    return frames, meta
