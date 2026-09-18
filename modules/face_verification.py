"""
modules.face_verification

Selfie-to-document face matching. Prefers deepface or face_recognition
(dlib-based 128-d embeddings) as the comparison backend, per
requirements.txt. deepface is the one this project ships and has been
tested with — it installs from prebuilt wheels (tensorflow + tf-keras),
unlike face_recognition's dlib dependency below.

IMPORTANT — environment note: face_recognition depends on dlib, which
ships no prebuilt wheel for every platform and can require a slow
from-source compile (needs cmake + a C++ toolchain). deepface pulls in
a full deep-learning stack (tensorflow/keras). When NEITHER is
installed, this module automatically falls back to a low-fidelity
OpenCV heuristic comparator (histogram + template correlation of the
aligned face crop) so the rest of the pipeline keeps working end to
end rather than hard-failing on a missing optional dependency — the
same "never crash" pattern used throughout this project (see
modules.ocr's tesseract/easyocr split). The fallback is clearly
flagged in every result it produces and should NOT be relied on for
real identity decisions; install face_recognition or deepface for
production use.

Face detection itself (for locating regions and counting faces, which
is needed regardless of comparison backend) always has a working
implementation via OpenCV's bundled Haar cascade, used when the
preferred backend's own detector isn't available.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import face_recognition as _face_recognition_lib
    _FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    _FACE_RECOGNITION_AVAILABLE = False

try:
    from deepface import DeepFace as _DeepFace
    _DEEPFACE_AVAILABLE = True
except ImportError:
    _DEEPFACE_AVAILABLE = False


# Display similarity is calibrated so 50.0 always corresponds to the
# backend's own match/no-match distance threshold — see
# _distance_to_similarity() below for why.
DISPLAY_MATCH_THRESHOLD = 50.0

# face_recognition's own documented default tolerance for "same person".
_FACE_RECOGNITION_MATCH_DISTANCE = 0.6

# Threshold used only by the OpenCV fallback comparator, in its own
# similarity-like units (not a distance) — kept separate from
# DISPLAY_MATCH_THRESHOLD since it's a fundamentally weaker, differently-
# shaped signal that shouldn't be presented as equivalent to a real
# embedding-distance comparison.
_FALLBACK_MATCH_THRESHOLD = 60.0

_FACE_CROP_SIZE = (150, 150)

_face_cascade: Optional[cv2.CascadeClassifier] = None  # lazy singleton


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_bgr(image: Any) -> np.ndarray:
    """Ensure a 3-channel BGR array, converting from grayscale if needed."""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _to_gray(image: Any) -> np.ndarray:
    """Ensure a single-channel grayscale array, converting from BGR if needed."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_rgb(image: Any) -> np.ndarray:
    """Convert to RGB (the channel order face_recognition/dlib expects)."""
    bgr = _to_bgr(image)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Lazily load OpenCV's bundled frontal-face Haar cascade."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _largest_face(faces: List[Dict[str, int]]) -> Dict[str, int]:
    """Pick the largest-area face bbox from a list — the usual heuristic for 'the subject'."""
    return max(faces, key=lambda f: f["width"] * f["height"])


def _crop_face(image: Any, bbox: Dict[str, int], margin: float = 0.15) -> np.ndarray:
    """Crop a face region out of an image with a small margin, clamped to image bounds."""
    h_img, w_img = image.shape[:2]
    left, top, width, height = bbox["left"], bbox["top"], bbox["width"], bbox["height"]
    mx, my = int(width * margin), int(height * margin)
    x0, y0 = max(0, left - mx), max(0, top - my)
    x1, y1 = min(w_img, left + width + mx), min(h_img, top + height + my)
    return image[y0:y1, x0:x1]


def _normalize_face_crop(crop: np.ndarray) -> np.ndarray:
    """Resize + grayscale + histogram-equalize a face crop for lighting-robust comparison."""
    gray = _to_gray(crop)
    resized = cv2.resize(gray, _FACE_CROP_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(resized)


def _bbox_to_face_recognition_location(bbox: Dict[str, int]) -> tuple:
    """Convert our {left, top, width, height} bbox to face_recognition's (top, right, bottom, left)."""
    left, top, width, height = bbox["left"], bbox["top"], bbox["width"], bbox["height"]
    return (top, left + width, top + height, left)


def _face_recognition_location_to_bbox(location: tuple) -> Dict[str, int]:
    """Convert face_recognition's (top, right, bottom, left) to our {left, top, width, height} bbox."""
    top, right, bottom, left = location
    return {"left": int(left), "top": int(top), "width": int(right - left), "height": int(bottom - top)}


def _distance_to_similarity(distance: float, match_distance_threshold: float) -> float:
    """
    Map a raw embedding distance to a 0-100 display similarity score,
    linearly calibrated so that distance == match_distance_threshold
    always lands exactly on DISPLAY_MATCH_THRESHOLD (50.0) — i.e. "50"
    always means "right at this backend's own match boundary", 100
    means identical encodings, 0 means distance >= 2x the threshold.

    This is an easy-to-reason-about *display* transform, not a
    calibrated probability of same-identity — raw distance-to-
    probability mapping is model-specific and isn't something these
    libraries expose directly.

    Args:
        distance (float): Raw embedding distance (lower = more similar).
        match_distance_threshold (float): The backend's own distance
            cutoff for "same person".

    Returns:
        float: Similarity score in [0, 100].
    """
    if match_distance_threshold <= 0:
        return 0.0
    ratio = distance / match_distance_threshold
    similarity = DISPLAY_MATCH_THRESHOLD * (2.0 - ratio)
    return round(max(0.0, min(100.0, similarity)), 1)


# ---------------------------------------------------------------------------
# Face detection (backend-aware, with an always-available OpenCV path)
# ---------------------------------------------------------------------------

def _detect_faces_opencv(image: Any) -> List[Dict[str, int]]:
    """Detect faces via OpenCV's Haar cascade. Always available, used as the universal fallback."""
    gray = _to_gray(image)
    faces = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [{"left": int(x), "top": int(y), "width": int(w), "height": int(h)} for (x, y, w, h) in faces]


def _detect_faces(image: Any, backend: str) -> List[Dict[str, int]]:
    """
    Detect all faces in an image using the requested backend's own
    detector when available, otherwise OpenCV's Haar cascade.

    Args:
        image: Input image (numpy array).
        backend (str): "face_recognition", "deepface", or anything
            else (treated as "use the OpenCV fallback").

    Returns:
        list[dict]: Face bounding boxes as {"left", "top", "width", "height"}.
    """
    if backend == "face_recognition" and _FACE_RECOGNITION_AVAILABLE:
        rgb = _to_rgb(image)
        locations = _face_recognition_lib.face_locations(rgb)
        return [_face_recognition_location_to_bbox(loc) for loc in locations]

    if backend == "deepface" and _DEEPFACE_AVAILABLE:
        bgr = _to_bgr(image)
        try:
            faces = _DeepFace.extract_faces(bgr, detector_backend="opencv", enforce_detection=False)
        except Exception:
            faces = []
        boxes = []
        for face in faces:
            area = face.get("facial_area", {})
            # DeepFace returns a placeholder full-image region with confidence 0
            # when detection finds nothing — skip those.
            if face.get("confidence", 1) and area.get("w", 0) > 0 and area.get("h", 0) > 0:
                boxes.append({"left": int(area["x"]), "top": int(area["y"]),
                              "width": int(area["w"]), "height": int(area["h"])})
        return boxes

    return _detect_faces_opencv(image)


def detect_face(image: Any, backend: str = "deepface") -> Optional[Dict[str, Any]]:
    """
    Detect the primary (largest) face in an image.

    Never raises — returns None on any detection failure or if no
    face is found.

    Args:
        image: Input image (document photo region or selfie).
        backend (str): "face_recognition", "deepface", or the OpenCV
            fallback is used automatically if the requested backend
            isn't installed.

    Returns:
        dict | None: {"bbox": {"left","top","width","height"}, "face_count": int},
            or None if no face is detected.
    """
    try:
        faces = _detect_faces(image, (backend or "face_recognition").lower())
    except Exception:
        return None
    if not faces:
        return None
    return {"bbox": _largest_face(faces), "face_count": len(faces)}


# ---------------------------------------------------------------------------
# Face encoding
# ---------------------------------------------------------------------------

def extract_face_encoding(image: Any, backend: str = "deepface") -> Optional[Any]:
    """
    Compute a face embedding/encoding for the largest detected face in
    an image, for use in similarity comparison.

    Never raises — returns None if no face is found or encoding fails.

    Args:
        image: Input image containing a face.
        backend (str): "face_recognition", "deepface", or the OpenCV
            fallback (a normalized-pixel vector stand-in — NOT a real
            biometric embedding) if neither library is installed.

    Returns:
        Any | None: Face embedding vector, or None if no face found.
    """
    backend = (backend or "deepface").lower()
    try:
        faces = _detect_faces(image, backend)
        if not faces:
            return None
        bbox = _largest_face(faces)

        if backend == "face_recognition" and _FACE_RECOGNITION_AVAILABLE:
            rgb = _to_rgb(image)
            encodings = _face_recognition_lib.face_encodings(rgb, known_face_locations=[_bbox_to_face_recognition_location(bbox)])
            return encodings[0] if encodings else None

        if backend == "deepface" and _DEEPFACE_AVAILABLE:
            crop = _crop_face(image, bbox)
            reps = _DeepFace.represent(_to_bgr(crop), model_name="VGG-Face", enforce_detection=False)
            return np.array(reps[0]["embedding"]) if reps else None

        # OpenCV fallback stand-in "encoding": a flattened, normalized,
        # lighting-equalized pixel vector of the aligned face crop.
        crop = _crop_face(image, bbox)
        normalized = _normalize_face_crop(crop)
        return normalized.flatten().astype(np.float32) / 255.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-backend comparison
# ---------------------------------------------------------------------------

def _compare_with_face_recognition(doc_image: Any, selfie_image: Any, doc_bbox: Dict, selfie_bbox: Dict) -> Dict[str, Any]:
    """Compare two face crops using face_recognition's dlib embeddings + Euclidean distance."""
    doc_rgb, selfie_rgb = _to_rgb(doc_image), _to_rgb(selfie_image)
    doc_encodings = _face_recognition_lib.face_encodings(doc_rgb, known_face_locations=[_bbox_to_face_recognition_location(doc_bbox)])
    selfie_encodings = _face_recognition_lib.face_encodings(selfie_rgb, known_face_locations=[_bbox_to_face_recognition_location(selfie_bbox)])

    if not doc_encodings or not selfie_encodings:
        return {
            "is_match": None, "similarity_score": None, "threshold": DISPLAY_MATCH_THRESHOLD,
            "backend_used": "face_recognition", "raw_distance": None,
            "notes": ["Could not compute a face encoding for one or both images."],
        }

    distance = float(_face_recognition_lib.face_distance([doc_encodings[0]], selfie_encodings[0])[0])
    similarity = _distance_to_similarity(distance, _FACE_RECOGNITION_MATCH_DISTANCE)
    return {
        "is_match": similarity >= DISPLAY_MATCH_THRESHOLD,
        "similarity_score": similarity,
        "threshold": DISPLAY_MATCH_THRESHOLD,
        "backend_used": "face_recognition",
        "raw_distance": round(distance, 4),
        "notes": [],
    }


def _compare_with_deepface(doc_image: Any, selfie_image: Any) -> Dict[str, Any]:
    """Compare two images using deepface's verify() (handles its own detection + embedding internally)."""
    verification = _DeepFace.verify(_to_bgr(doc_image), _to_bgr(selfie_image), model_name="VGG-Face", enforce_detection=False)
    distance = float(verification.get("distance", 1.0))
    model_threshold = float(verification.get("threshold") or 0.4) or 0.4
    similarity = _distance_to_similarity(distance, model_threshold)
    return {
        "is_match": bool(verification.get("verified", similarity >= DISPLAY_MATCH_THRESHOLD)),
        "similarity_score": similarity,
        "threshold": DISPLAY_MATCH_THRESHOLD,
        "backend_used": "deepface",
        "raw_distance": round(distance, 4),
        "notes": [],
    }


def _compare_with_opencv_fallback(doc_image: Any, selfie_image: Any, doc_bbox: Dict, selfie_bbox: Dict) -> Dict[str, Any]:
    """
    Low-fidelity comparator used only when neither face_recognition nor
    deepface is installed: histogram correlation + normalized template
    matching on the aligned, lighting-equalized face crops. This is a
    much weaker signal than a real biometric embedding and should not
    be relied on for real identity decisions.
    """
    doc_norm = _normalize_face_crop(_crop_face(doc_image, doc_bbox))
    selfie_norm = _normalize_face_crop(_crop_face(selfie_image, selfie_bbox))

    doc_hist = cv2.calcHist([doc_norm], [0], None, [256], [0, 256])
    selfie_hist = cv2.calcHist([selfie_norm], [0], None, [256], [0, 256])
    cv2.normalize(doc_hist, doc_hist)
    cv2.normalize(selfie_hist, selfie_hist)
    hist_correlation = float(cv2.compareHist(doc_hist, selfie_hist, cv2.HISTCMP_CORREL))

    template_score = float(cv2.matchTemplate(doc_norm, selfie_norm, cv2.TM_CCOEFF_NORMED)[0][0])

    combined = 0.5 * hist_correlation + 0.5 * template_score  # roughly in [-1, 1]
    similarity = round(max(0.0, min(1.0, (combined + 1.0) / 2.0)) * 100, 1)

    return {
        "is_match": similarity >= _FALLBACK_MATCH_THRESHOLD,
        "similarity_score": similarity,
        "threshold": _FALLBACK_MATCH_THRESHOLD,
        "backend_used": "opencv_fallback",
        "raw_distance": None,
        "notes": [
            "Neither face_recognition nor deepface is installed in this environment; used a low-fidelity "
            "OpenCV histogram/template-correlation proxy instead of true face-embedding comparison. "
            "This fallback is far less reliable and should not be used for real identity decisions — "
            "install face_recognition or deepface for production use."
        ],
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compare_faces(document_face_image: Any, selfie_image: Any, backend: str = "deepface") -> Dict[str, Any]:
    """
    Compare the face in a document photo against a live/uploaded
    selfie and return a similarity score and match/no-match verdict.

    Handles edge cases explicitly rather than crashing or guessing
    silently:
        - Missing image(s): returns a null verdict with an explanatory note.
        - No face detected in either image: returns a null verdict
          (is_match=None) — a comparison cannot be meaningfully made.
        - Multiple faces detected in either image: proceeds using the
          largest face in each (the common "primary subject" heuristic),
          but flags this prominently via multiple_faces_flagged and a
          note, since the largest face might not be the actual subject.

    Never raises: any backend failure is caught and reported via the
    "error" field with a null verdict, rather than propagating.

    Args:
        document_face_image: Document image (or a crop containing the
            photo) — face detection locates the photo region itself,
            so the full document page image works fine.
        selfie_image: User-supplied selfie or live-capture image.
        backend (str): "face_recognition" or "deepface". Falls back
            automatically (with a note) to a low-fidelity OpenCV
            comparator if the requested library isn't installed.

    Returns:
        dict: {
            "is_match": bool | None,           # None if no verdict could be reached
            "similarity_score": float | None,  # 0-100
            "threshold": float,                # match cutoff, in the same 0-100 units
            "backend_used": str,
            "raw_distance": float | None,       # backend-native distance, where applicable
            "document_face_count": int,
            "selfie_face_count": int,
            "multiple_faces_flagged": bool,
            "no_face_detected": bool,
            "error": str | None,
            "notes": list[str],
        }
    """
    backend = (backend or "deepface").lower()
    notes: List[str] = []

    if document_face_image is None or selfie_image is None:
        return {
            "is_match": None, "similarity_score": None, "threshold": DISPLAY_MATCH_THRESHOLD,
            "backend_used": backend, "raw_distance": None,
            "document_face_count": 0, "selfie_face_count": 0,
            "multiple_faces_flagged": False, "no_face_detected": True,
            "error": None, "notes": ["Both a document photo and a selfie/live photo are required."],
        }

    try:
        doc_faces = _detect_faces(document_face_image, backend)
        selfie_faces = _detect_faces(selfie_image, backend)
    except Exception as exc:
        return {
            "is_match": None, "similarity_score": None, "threshold": DISPLAY_MATCH_THRESHOLD,
            "backend_used": backend, "raw_distance": None,
            "document_face_count": 0, "selfie_face_count": 0,
            "multiple_faces_flagged": False, "no_face_detected": False,
            "error": f"Face detection failed: {exc}", "notes": [],
        }

    doc_count, selfie_count = len(doc_faces), len(selfie_faces)

    if doc_count == 0:
        notes.append("No face detected in the document photo.")
    if selfie_count == 0:
        notes.append("No face detected in the selfie/live photo.")
    if doc_count == 0 or selfie_count == 0:
        return {
            "is_match": None, "similarity_score": None, "threshold": DISPLAY_MATCH_THRESHOLD,
            "backend_used": backend, "raw_distance": None,
            "document_face_count": doc_count, "selfie_face_count": selfie_count,
            "multiple_faces_flagged": False, "no_face_detected": True,
            "error": None, "notes": notes,
        }

    multiple_faces_flagged = False
    if doc_count > 1:
        notes.append(f"Multiple faces ({doc_count}) detected in the document photo; used the largest face region.")
        multiple_faces_flagged = True
    if selfie_count > 1:
        notes.append(f"Multiple faces ({selfie_count}) detected in the selfie; used the largest face region.")
        multiple_faces_flagged = True

    doc_bbox, selfie_bbox = _largest_face(doc_faces), _largest_face(selfie_faces)

    if backend in ("face_recognition", "deepface") and not (
        (backend == "face_recognition" and _FACE_RECOGNITION_AVAILABLE)
        or (backend == "deepface" and _DEEPFACE_AVAILABLE)
    ):
        notes.append(f"'{backend}' is not installed in this environment; used the OpenCV fallback comparator instead.")

    try:
        if backend == "face_recognition" and _FACE_RECOGNITION_AVAILABLE:
            result = _compare_with_face_recognition(document_face_image, selfie_image, doc_bbox, selfie_bbox)
        elif backend == "deepface" and _DEEPFACE_AVAILABLE:
            result = _compare_with_deepface(document_face_image, selfie_image)
        else:
            result = _compare_with_opencv_fallback(document_face_image, selfie_image, doc_bbox, selfie_bbox)
    except Exception as exc:
        return {
            "is_match": None, "similarity_score": None, "threshold": DISPLAY_MATCH_THRESHOLD,
            "backend_used": backend, "raw_distance": None,
            "document_face_count": doc_count, "selfie_face_count": selfie_count,
            "multiple_faces_flagged": multiple_faces_flagged, "no_face_detected": False,
            "error": f"Face comparison failed: {exc}", "notes": notes,
        }

    result["document_face_count"] = doc_count
    result["selfie_face_count"] = selfie_count
    result["multiple_faces_flagged"] = multiple_faces_flagged
    result["no_face_detected"] = False
    result["notes"] = notes + result.get("notes", [])
    result["error"] = None
    return result


# ---------------------------------------------------------------------------
# Not yet implemented (deferred)
# ---------------------------------------------------------------------------

def check_liveness(image: Any) -> Dict[str, Any]:
    """
    Run basic liveness/spoof checks on a selfie image (e.g. detecting
    printed photo or screen replay artifacts).

    Not yet implemented — deferred; not part of this module's initial
    build (face detection, encoding, and comparison).

    Args:
        image: Selfie image.

    Returns:
        dict: e.g. {"is_live": bool, "confidence": float, "notes": [...]}
    """
    pass


def assess_face_image_quality(image: Any) -> Dict[str, Any]:
    """
    Assess whether a face image meets minimum quality requirements
    for reliable comparison (blur, lighting, occlusion, pose angle).

    Not yet implemented — deferred; not part of this module's initial
    build (face detection, encoding, and comparison).

    Args:
        image: Face image (document or selfie).

    Returns:
        dict: e.g. {"is_acceptable": bool, "issues": [...]}
    """
    pass
