"""
utils.preprocessing

Image loading and preprocessing helpers shared across the OCR,
tampering-detection, and face-verification modules.

Implemented so far:
    - validate_upload / save_temp_file  (upload handling)
    - load_image / get_image_dimensions (I/O)
    - convert_to_grayscale / denoise_image / deskew_image /
      enhance_contrast (individual OCR-prep stages, OpenCV-backed)
    - preprocess_for_ocr / run_preprocessing_pipeline (orchestration,
      the latter also returning each intermediate stage for a
      before/after UI preview)
    - to_display_image (numpy BGR/gray -> RGB for st.image)

Still placeholder (filled in by later prompts):
    - normalize_orientation (EXIF-based rotation correction)
    - detect_document_edges / crop_and_warp_document (perspective
      correction for photographed, non-flatbed documents)
    - resize_image is implemented (used internally by the pipeline
      and exposed directly for callers that need it standalone)
"""

import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Upload validation & temp-file handling
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE_MB = 10.0


def validate_upload(
    uploaded_file: Any,
    allowed_extensions: Optional[set] = None,
    max_size_mb: float = MAX_FILE_SIZE_MB,
) -> Dict[str, Any]:
    """
    Validate an uploaded file's extension and size before it enters
    the processing pipeline.

    Args:
        uploaded_file: Streamlit UploadedFile (or any file-like object
            exposing `.name` and either `.size` or a seekable stream).
        allowed_extensions (set | None): Allowed extensions (without
            the dot), defaults to ALLOWED_EXTENSIONS.
        max_size_mb (float): Maximum allowed file size in megabytes.

    Returns:
        dict: {
            "is_valid": bool,
            "errors": list[str],
            "extension": str,
            "size_mb": float,
        }
    """
    allowed = allowed_extensions or ALLOWED_EXTENSIONS
    errors = []

    name = getattr(uploaded_file, "name", "") or ""
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if not extension:
        errors.append("Could not determine file type — no file extension found.")
    elif extension not in allowed:
        errors.append(
            f"Unsupported file type '.{extension}'. Allowed types: "
            f"{', '.join(sorted(allowed))}."
        )

    size_bytes = getattr(uploaded_file, "size", None)
    if size_bytes is None:
        try:
            pos = uploaded_file.tell()
            uploaded_file.seek(0, os.SEEK_END)
            size_bytes = uploaded_file.tell()
            uploaded_file.seek(pos)
        except Exception:
            size_bytes = 0

    size_mb = round(size_bytes / (1024 * 1024), 2) if size_bytes else 0.0
    if size_mb > max_size_mb:
        errors.append(f"File too large ({size_mb} MB). Maximum allowed is {max_size_mb} MB.")
    if size_bytes == 0:
        errors.append("Uploaded file appears to be empty.")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "extension": extension,
        "size_mb": size_mb,
    }


def save_temp_file(uploaded_file: Any, suffix: Optional[str] = None) -> str:
    """
    Save an uploaded file's bytes to a temporary path on disk.

    Args:
        uploaded_file: Streamlit UploadedFile (or any file-like object
            with `.read()`/`.seek()`).
        suffix (str | None): Filename suffix to use for the temp file
            (e.g. ".jpg"). Inferred from the upload's name if omitted.

    Returns:
        str: Filesystem path to the saved temporary file.
    """
    name = getattr(uploaded_file, "name", "") or "upload"
    if suffix is None:
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ".png"

    uploaded_file.seek(0)
    data = uploaded_file.read()
    uploaded_file.seek(0)

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="veriscan_")
    with os.fdopen(fd, "wb") as f:
        f.write(data)

    return path


# ---------------------------------------------------------------------------
# Loading / basic I/O
# ---------------------------------------------------------------------------

def load_image(file_or_path: Any) -> np.ndarray:
    """
    Load an image from an uploaded file object, a filesystem path, a
    PIL Image, or an existing numpy array into a standard in-memory
    BGR numpy array (OpenCV convention).

    Args:
        file_or_path: Streamlit UploadedFile, file-like object,
            filesystem path string, PIL.Image.Image, or numpy array.

    Returns:
        np.ndarray: Loaded image as a BGR (or grayscale) numpy array.

    Raises:
        ValueError: If the image data could not be decoded.
        TypeError: If the input type is unsupported.
    """
    if isinstance(file_or_path, np.ndarray):
        return file_or_path

    if isinstance(file_or_path, Image.Image):
        rgb = np.array(file_or_path.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(file_or_path, str):
        image = cv2.imread(file_or_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image at path: {file_or_path}")
        return image

    if hasattr(file_or_path, "read"):
        try:
            file_or_path.seek(0)
        except Exception:
            pass
        data = file_or_path.read()
        try:
            file_or_path.seek(0)
        except Exception:
            pass
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode uploaded image data.")
        return image

    raise TypeError(f"Unsupported input type for load_image: {type(file_or_path)!r}")


def get_image_dimensions(image: Any) -> Tuple[int, int]:
    """
    Return the (width, height) of an image in pixels.

    Args:
        image: Input image (numpy array).

    Returns:
        tuple[int, int]: (width, height).
    """
    h, w = image.shape[:2]
    return (w, h)


def to_display_image(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR (or grayscale) numpy array as used internally by
    OpenCV into an RGB array suitable for st.image / PIL display.

    Args:
        image: Input image (numpy array), BGR or single-channel.

    Returns:
        np.ndarray: RGB (or unchanged grayscale) image for display.
    """
    if image is None:
        return image
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Individual preprocessing stages
# ---------------------------------------------------------------------------

def resize_image(image: Any, max_dimension: int = 1600) -> Any:
    """
    Resize an image so its largest dimension does not exceed
    max_dimension, preserving aspect ratio. Images already within the
    limit are returned unchanged.

    Args:
        image: Input image (numpy array).
        max_dimension (int): Maximum allowed width/height in pixels.

    Returns:
        np.ndarray: Resized image.
    """
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dimension:
        return image
    scale = max_dimension / float(largest)
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def normalize_orientation(image: np.ndarray) -> np.ndarray:
    """
    Ensure image orientation is valid, preserving original aspect ratio
    and portrait/landscape layout. Avoids forcibly rotating vertical/portrait images.
    """
    if image is None:
        return image
    return image



def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale. No-op if already single-channel.

    Args:
        image: Input image (numpy array).

    Returns:
        np.ndarray: Grayscale image.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Apply non-local-means denoising to improve downstream OCR and
    tampering-detection accuracy. Works on both grayscale and color
    images.

    Args:
        image: Input image (numpy array).

    Returns:
        np.ndarray: Denoised image.
    """
    if len(image.shape) == 2:
        return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return cv2.fastNlMeansDenoisingColored(
        image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct small rotational skew (as opposed to full
    perspective distortion — see detect_document_edges/
    crop_and_warp_document for that) using a minimum-area-rectangle
    fit over thresholded foreground pixels.

    Args:
        image: Input image (numpy array), grayscale or color.

    Returns:
        np.ndarray: Deskewed image, same shape/channels as input. If
            no foreground pixels are found, the image is returned
            unchanged.
    """
    gray = convert_to_grayscale(image)
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip correction for negligible skew to avoid needless resampling.
    if abs(angle) < 0.1:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance local contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalization). Applied to the luminance channel for
    color images to avoid color shifts.

    Args:
        image: Input image (numpy array), grayscale or color.

    Returns:
        np.ndarray: Contrast-enhanced image, same shape as input.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if len(image.shape) == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Perspective correction (not yet implemented)
# ---------------------------------------------------------------------------

def detect_document_edges(image: Any) -> Optional[Any]:
    """
    Detect the document boundary within a photographed image (as
    opposed to a flatbed scan), for later perspective correction.

    Args:
        image: Input image.

    Returns:
        Any | None: Detected contour/corner points, or None if not found.
    """
    pass


def crop_and_warp_document(image: Any, corners: Any) -> Any:
    """
    Apply a perspective transform to crop and flatten a document
    region given its detected corner points.

    Args:
        image: Input image.
        corners: Corner points from detect_document_edges.

    Returns:
        Any: Cropped and perspective-corrected document image.
    """
    pass


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def preprocess_for_ocr(image: np.ndarray, max_dimension: int = 1600) -> np.ndarray:
    """
    Run the full preprocessing pipeline tailored for OCR accuracy:
    orientation correction -> resize -> grayscale -> denoise -> deskew -> contrast enhancement.
    """
    oriented = normalize_orientation(image)
    resized = resize_image(oriented, max_dimension=max_dimension)
    gray = convert_to_grayscale(resized)
    denoised = denoise_image(gray)
    deskewed = deskew_image(denoised)
    enhanced = enhance_contrast(deskewed)
    return enhanced


def run_preprocessing_pipeline(image: np.ndarray, max_dimension: int = 1600) -> Dict[str, np.ndarray]:
    """
    Run the same stages as preprocess_for_ocr, but return every
    intermediate result keyed by stage name — intended for driving a
    before/after preview in the UI.
    """
    oriented = normalize_orientation(image)
    resized = resize_image(oriented, max_dimension=max_dimension)
    grayscale = convert_to_grayscale(resized)
    denoised = denoise_image(grayscale)
    deskewed = deskew_image(denoised)
    contrast_enhanced = enhance_contrast(deskewed)

    return {
        "original": image,
        "resized": resized,
        "grayscale": grayscale,
        "denoised": denoised,
        "deskewed": deskewed,
        "contrast_enhanced": contrast_enhanced,
    }

