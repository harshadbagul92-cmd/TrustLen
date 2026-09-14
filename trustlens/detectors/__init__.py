"""Detection lanes. Each returns the same Evidence object."""
from ..schemas import Modality
from .audio import AudioDetector
from .image import ImageDetector
from .text import TextDetector
from .video import VideoDetector

_TEXT, _IMAGE, _AUDIO, _VIDEO = TextDetector(), ImageDetector(), AudioDetector(), VideoDetector()

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".wma")
VIDEO_EXT = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg")


def detect_modality(filename: str = "", content_type: str = "") -> Modality:
    """Route on content type first, then extension. Never on the rest of the
    filename - a file called 'deepfake.jpg' must be treated like any other."""
    ct = (content_type or "").lower()
    if ct.startswith("image/"):
        return Modality.IMAGE
    if ct.startswith("audio/"):
        return Modality.AUDIO
    if ct.startswith("video/"):
        return Modality.VIDEO
    if ct.startswith("text/"):
        return Modality.TEXT

    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in (filename or "") else ""
    if ext in IMAGE_EXT:
        return Modality.IMAGE
    if ext in AUDIO_EXT:
        return Modality.AUDIO
    if ext in VIDEO_EXT:
        return Modality.VIDEO
    return Modality.TEXT


def get(modality: Modality):
    return {Modality.TEXT: _TEXT, Modality.IMAGE: _IMAGE,
            Modality.AUDIO: _AUDIO, Modality.VIDEO: _VIDEO}[modality]


def analyse(data, filename: str = "", content_type: str = ""):
    """Route and analyse in one call. Used by both the API and the UI."""
    modality = detect_modality(filename, content_type)
    return get(modality).analyse(data, {"filename": filename, "content_type": content_type})


__all__ = ["TextDetector", "ImageDetector", "AudioDetector", "VideoDetector",
           "detect_modality", "get", "analyse"]
