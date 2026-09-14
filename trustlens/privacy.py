"""
Nothing the user uploads is kept.

TrustLens processes uploads in memory. The one unavoidable exception is
video: OpenCV's VideoCapture needs a real path on disk and cannot read from
a bytes buffer. So a video is written to a scratch file, read, and deleted -
deleted in a `finally`, so it goes even if the analysis raises.

This module is the single place that is allowed to touch the filesystem with
user content, which makes the guarantee auditable: grep for `scratch_file`
and you have found every write. `tests/` asserts the file is gone afterwards.
"""
import atexit
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from typing import Iterator, List

_ROOT = os.path.join(tempfile.gettempdir(), "trustlens-scratch")
_lock = threading.Lock()

# Purely a counter for the UI's privacy panel. It records how many scratch
# files have been created and removed. It never records names or content.
_created = 0
_removed = 0


def _ensure_root() -> str:
    os.makedirs(_ROOT, exist_ok=True)
    return _ROOT


def _shred(path: str) -> None:
    """
    Overwrite before unlinking, so the bytes are not trivially recoverable
    from free space. Best effort - on a journalling or copy-on-write
    filesystem this is not a guarantee, and we do not claim it is.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "r+b", buffering=0) as fh:
            fh.write(b"\0" * min(size, 1024 * 1024))
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        pass


@contextmanager
def scratch_file(data: bytes, suffix: str = "") -> Iterator[str]:
    """
    Write `data` to a temporary file, yield its path, then destroy it.

        with scratch_file(video_bytes, ".mp4") as path:
            cap = cv2.VideoCapture(path)

    The file is removed on the way out whether the block succeeds or raises.
    """
    global _created, _removed
    _ensure_root()
    path = os.path.join(_ROOT, f"{uuid.uuid4().hex}{suffix}")
    with _lock:
        _created += 1
    try:
        with open(path, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        yield path
    finally:
        if os.path.exists(path):
            _shred(path)
            try:
                os.remove(path)
                with _lock:
                    _removed += 1
            except Exception:
                # Windows can hold a lock briefly after a reader closes.
                # Leave it for purge_all() / atexit rather than raising.
                pass


def purge() -> int:
    """Remove anything left in the scratch directory. Returns the count."""
    global _removed
    if not os.path.isdir(_ROOT):
        return 0
    n = 0
    for name in os.listdir(_ROOT):
        full = os.path.join(_ROOT, name)
        try:
            if os.path.isfile(full):
                _shred(full)
                os.remove(full)
                n += 1
            else:
                shutil.rmtree(full, ignore_errors=True)
        except Exception:
            continue
    with _lock:
        _removed += n
    return n


def residue() -> List[str]:
    """Files still on disk. Should always be empty between analyses."""
    if not os.path.isdir(_ROOT):
        return []
    return [f for f in os.listdir(_ROOT)
            if os.path.isfile(os.path.join(_ROOT, f))]


def stats() -> dict:
    """For the UI privacy panel. Counts only - never names, never content."""
    return {
        "scratch_dir": _ROOT,
        "files_created": _created,
        "files_removed": _removed,
        "files_remaining": len(residue()),
    }


atexit.register(purge)
