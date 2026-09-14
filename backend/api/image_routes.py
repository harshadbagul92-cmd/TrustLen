import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from backend.services.image_analysis import ImageAnalysisService
from backend.schemas.detector_result import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["Image Authenticity"])
analysis_service = ImageAnalysisService()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/image", response_model=APIResponse)
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    POST /analyze/image
    Accepts an uploaded image (JPG/PNG/WEBP), runs pre-processing, metadata/provenance check,
    face/region detection, Hugging Face AI detector, evidence engine, and confidence engine.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded. Please provide an image file."
        )

    filename = file.filename or "upload.jpg"
    fn_lower = filename.lower()

    # 1. Extension Validation
    if not any(fn_lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension for '{filename}'. Allowed formats: JPG, JPEG, PNG, WEBP."
        )

    # 2. MIME Type Validation
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_MIME_TYPES and content_type != "application/octet-stream":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MIME type '{content_type}'. Allowed types: image/jpeg, image/png, image/webp."
        )

    # 3. Read File Bytes in Memory (Ephemeral, never stored permanently)
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file stream: {str(e)}"
        )

    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds max limit of 20 MB."
        )

    # 4. Pipeline Execution & Clean Error Handling
    try:
        result = analysis_service.analyze_image(file_bytes, filename=filename)
        return APIResponse(success=True, result=result)

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as exc:
        logger.error(f"[Image Route Exception] {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image analysis pipeline failure: {str(exc)}"
        )
