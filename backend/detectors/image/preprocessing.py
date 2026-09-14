import hashlib
import io
from typing import Dict, Any, Tuple
from PIL import Image, ExifTags


class ImagePreprocessingResult:
    def __init__(
        self,
        image: Image.Image,
        sha256: str,
        width: int,
        height: int,
        format_name: str,
        mode: str,
        size_bytes: int,
        exif_raw: Dict[str, Any],
        normalized_image: Image.Image
    ):
        self.image = image
        self.sha256 = sha256
        self.width = width
        self.height = height
        self.format_name = format_name
        self.mode = mode
        self.size_bytes = size_bytes
        self.exif_raw = exif_raw
        self.normalized_image = normalized_image


class ImagePreprocessor:
    """
    Safely inspects, validates, hashes, extracts EXIF, and normalizes image uploads.
    """

    ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}
    MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB max

    def preprocess(self, image_bytes: bytes, filename: str = "upload.jpg") -> ImagePreprocessingResult:
        if not image_bytes:
            raise ValueError("Empty image byte stream received.")

        size_bytes = len(image_bytes)
        if size_bytes > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size ({size_bytes / (1024*1024):.1f} MB) exceeds maximum limit of 20 MB.")

        # SHA-256 Hash
        sha256_hash = hashlib.sha256(image_bytes).hexdigest()

        # Open image safely with Pillow
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()  # Verify integrity
        except Exception as e:
            raise ValueError(f"Corrupted or undecodable image file: {str(e)}")

        # Re-open after verify() per Pillow documentation
        pil_img = Image.open(io.BytesIO(image_bytes))

        width, height = pil_img.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        fmt = (pil_img.format or "JPEG").upper()
        if fmt == "JPG":
            fmt = "JPEG"

        # EXIF Extraction
        exif_dict = {}
        try:
            raw_exif = pil_img._getexif()
            if raw_exif:
                for tag_id, val in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Avoid non-serializable raw bytes in dict values
                    if isinstance(val, bytes):
                        exif_dict[str(tag_name)] = val.hex()[:64]
                    else:
                        exif_dict[str(tag_name)] = str(val)
        except Exception:
            exif_dict = {}

        # Create normalized RGB image copy (224x224 or preserved aspect ratio RGB)
        normalized_img = pil_img.convert("RGB")

        return ImagePreprocessingResult(
            image=pil_img,
            sha256=sha256_hash,
            width=width,
            height=height,
            format_name=fmt,
            mode=pil_img.mode,
            size_bytes=size_bytes,
            exif_raw=exif_dict,
            normalized_image=normalized_img
        )
