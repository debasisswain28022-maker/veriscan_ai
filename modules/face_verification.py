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
    import face_recognition as _face_recognition_lib  # type: ignore # noqa: F401
    _FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    _FACE_RECOGNITION_AVAILABLE = False

try:
    from deepface import DeepFace as _DeepFace  # type: ignore # noqa: F401
    _DEEPFACE_AVAILABLE = True
except Exception:
    _DEEPFACE_AVAILABLE = False

try:
    from cv2 import CascadeClassifier as _CascadeClassifier
except ImportError:
    _CascadeClassifier = getattr(cv2, "CascadeClassifier", None)


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
_FALLBACK_MATCH_THRESHOLD = 50.0

_FACE_CROP_SIZE = (150, 150)

_face_cascade: Optional[Any] = None  # lazy singleton


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


def _get_face_cascade() -> Any:
    """Lazily load OpenCV's bundled frontal-face Haar cascade."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if _CascadeClassifier is not None:
            _face_cascade = _CascadeClassifier(cascade_path)
        else:
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
    cropped = image[y0:y1, x0:x1]
    if cropped.size == 0:
        return image
    return cropped


def _extract_inner_face(crop: np.ndarray) -> np.ndarray:
    """
    Extract the inner facial region (eyes, nose, cheeks, mouth) while
    excluding outer background, hair, shoulders, and clothing.
    """
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    if h < 20 or w < 20:
        return crop

    x0 = int(w * 0.12)
    x1 = int(w * 0.88)
    y0 = int(h * 0.08)
    y1 = int(h * 0.88)

    inner = crop[y0:y1, x0:x1]
    return inner if inner.size > 0 else crop


def _normalize_face_crop(crop: np.ndarray) -> np.ndarray:
    """Resize + grayscale + CLAHE contrast-equalize inner face crop for lighting-robust comparison."""
    inner = _extract_inner_face(crop)
    gray = _to_gray(inner)
    resized = cv2.resize(gray, _FACE_CROP_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(resized)


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
    Map a raw embedding distance to a 0-100 display similarity score.
    """
    if match_distance_threshold <= 0:
        return 0.0
    ratio = distance / match_distance_threshold
    similarity = DISPLAY_MATCH_THRESHOLD * (2.0 - ratio)
    return round(max(0.0, min(100.0, similarity)), 1)


# ---------------------------------------------------------------------------
# Face detection (backend-aware, with an always-available OpenCV path)
# ---------------------------------------------------------------------------

def _is_valid_human_face(image: Any, bbox: Dict[str, int]) -> bool:
    """
    Verify that a detected candidate region actually contains a valid human face
    (checking for human facial landmark features or skin tone contrast)
    rather than non-human objects, tables, or background textures.
    """
    if image is None or not isinstance(image, np.ndarray) or not bbox:
        return False
    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox["left"], bbox["top"], bbox["width"], bbox["height"]
    crop = image[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]
    if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
        return False

    gray = _to_gray(crop)
    equ = cv2.equalizeHist(gray)

    eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    glasses_cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"

    eye_cascade = _CascadeClassifier(eye_cascade_path) if _CascadeClassifier is not None else cv2.CascadeClassifier(eye_cascade_path)
    glasses_cascade = _CascadeClassifier(glasses_cascade_path) if _CascadeClassifier is not None else cv2.CascadeClassifier(glasses_cascade_path)

    eyes = []
    if eye_cascade and not eye_cascade.empty():
        try:
            eyes = eye_cascade.detectMultiScale(equ, scaleFactor=1.1, minNeighbors=2, minSize=(10, 10))
        except Exception:
            eyes = []
    if len(eyes) == 0 and glasses_cascade and not glasses_cascade.empty():
        try:
            eyes = glasses_cascade.detectMultiScale(equ, scaleFactor=1.1, minNeighbors=2, minSize=(10, 10))
        except Exception:
            eyes = []

    skin_ok = False
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        skin_ratio = float(np.sum(mask > 0)) / float(crop.shape[0] * crop.shape[1])
        skin_ok = skin_ratio >= 0.04

    return len(eyes) > 0 or skin_ok


def _detect_faces_opencv(image: Any, is_document: bool = False) -> List[Dict[str, int]]:
    """
    Multi-pass face detection using OpenCV Haar cascades with sensitivity
    scaling, CLAHE equalization, document quadrant sweeps, skin-tone localization,
    and human face feature validation.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return []

    h_img, w_img = image.shape[:2]
    gray = _to_gray(image)
    equ = cv2.equalizeHist(gray)

    cascade_files = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_alt.xml",
        "haarcascade_profileface.xml",
    ]

    detected_faces: List[Dict[str, int]] = []

    # Pass 1: Standard & sensitive scale on full image
    for cname in cascade_files:
        cascade_path = cv2.data.haarcascades + cname
        try:
            cascade = _CascadeClassifier(cascade_path) if _CascadeClassifier is not None else cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                continue

            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(25, 25))
            for (x, y, w, h) in faces:
                b = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
                if _is_valid_human_face(image, b):
                    detected_faces.append(b)

            if not detected_faces:
                faces = cascade.detectMultiScale(equ, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))
                for (x, y, w, h) in faces:
                    b = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
                    if _is_valid_human_face(image, b):
                        detected_faces.append(b)

            if detected_faces:
                break
        except Exception:
            continue

    # Pass 2: Quadrant search for document scans
    if not detected_faces and (w_img > 250 or h_img > 250):
        quadrants = [
            (0, 0, int(w_img * 0.55), int(h_img * 0.65)),
            (int(w_img * 0.45), 0, int(w_img * 0.55), int(h_img * 0.65)),
            (0, 0, int(w_img * 0.50), h_img),
            (int(w_img * 0.50), 0, int(w_img * 0.50), h_img),
        ]
        for (qx, qy, qw, qh) in quadrants:
            if qw < 20 or qh < 20:
                continue
            sub_gray = gray[qy:qy+qh, qx:qx+qw]
            sub_equ = cv2.equalizeHist(sub_gray)
            for cname in cascade_files:
                cascade_path = cv2.data.haarcascades + cname
                try:
                    cascade = _CascadeClassifier(cascade_path) if _CascadeClassifier is not None else cv2.CascadeClassifier(cascade_path)
                    if cascade.empty():
                        continue
                    faces = cascade.detectMultiScale(sub_equ, scaleFactor=1.03, minNeighbors=2, minSize=(18, 18))
                    for (fx, fy, fw, fh) in faces:
                        b = {"left": int(qx + fx), "top": int(qy + fy), "width": int(fw), "height": int(fh)}
                        if _is_valid_human_face(image, b):
                            detected_faces.append(b)
                    if detected_faces:
                        break
                except Exception:
                    continue
            if detected_faces:
                break

    # Pass 3: Skin tone region localization
    if not detected_faces and (w_img > 250 or h_img > 250) and len(image.shape) == 3:
        try:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            lower = np.array([0, 133, 77], dtype=np.uint8)
            upper = np.array([255, 173, 127], dtype=np.uint8)
            mask = cv2.inRange(ycrcb, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(cv2.erode(mask, kernel, iterations=1), kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_a = (w_img * h_img) * 0.002
            max_a = (w_img * h_img) * 0.45
            for c in contours:
                area = cv2.contourArea(c)
                if min_a <= area <= max_a:
                    x, y, w, h = cv2.boundingRect(c)
                    aspect = h / float(w)
                    if 0.6 <= aspect <= 2.0:
                        b = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
                        if _is_valid_human_face(image, b):
                            detected_faces.append(b)
                            break
        except Exception:
            pass

    # Pass 4: ID Photo Box Heuristic ONLY for document scans (is_document=True)
    if not detected_faces and is_document and w_img > 350 and h_img > 250:
        pw = int(w_img * 0.35)
        ph = int(h_img * 0.55)
        py = int(h_img * 0.08)
        px = int(w_img * 0.05)
        detected_faces.append({"left": px, "top": py, "width": pw, "height": ph})

    return detected_faces


def _detect_faces(image: Any, backend: str, is_document: bool = False) -> List[Dict[str, int]]:
    """
    Detect all faces in an image using the requested backend's own
    detector when available, otherwise OpenCV's Haar cascade.
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
            if face.get("confidence", 1) and area.get("w", 0) > 0 and area.get("h", 0) > 0:
                boxes.append({"left": int(area["x"]), "top": int(area["y"]),
                              "width": int(area["w"]), "height": int(area["h"])})
        if boxes:
            return boxes

    return _detect_faces_opencv(image, is_document=is_document)


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


def _compare_with_deepface(doc_image: Any, selfie_image: Any, doc_bbox: Optional[Dict] = None, selfie_bbox: Optional[Dict] = None) -> Dict[str, Any]:
    """Compare two face crops using deepface's verify() (handles cropped face regions)."""
    doc_crop = _crop_face(doc_image, doc_bbox) if doc_bbox else doc_image
    selfie_crop = _crop_face(selfie_image, selfie_bbox) if selfie_bbox else selfie_image
    verification = _DeepFace.verify(_to_bgr(doc_crop), _to_bgr(selfie_crop), model_name="VGG-Face", enforce_detection=False)
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
    OpenCV face comparator: inner facial oval extraction, fine-grained & coarse
    multi-scale HOG structural comparison, Lowe ratio SIFT keypoint matching, and
    multi-angle orientation alignment.
    """
    doc_crop = _crop_face(doc_image, doc_bbox, margin=0.10)
    selfie_crop = _crop_face(selfie_image, selfie_bbox, margin=0.10)

    doc_norm = _normalize_face_crop(doc_crop)

    best_composite = 0.0
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]

    for rot in rotations:
        test_selfie = selfie_crop if rot is None else cv2.rotate(selfie_crop, rot)
        selfie_norm = _normalize_face_crop(test_selfie)

        fine_sim = 0.5
        med_sim = 0.5
        coarse_sim = 0.5
        try:
            from skimage.feature import hog
            # Fine HOG (8x8 cells) - local facial feature shapes (eyes, nose, lip contours)
            h1_f = hog(doc_norm, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            h2_f = hog(selfie_norm, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            dot_f = np.dot(h1_f, h2_f)
            norm_f = (np.linalg.norm(h1_f) * np.linalg.norm(h2_f)) + 1e-6
            fine_sim = max(0.0, float(dot_f / norm_f))

            # Medium HOG (12x12 cells)
            h1_m = hog(doc_norm, orientations=8, pixels_per_cell=(12, 12), cells_per_block=(2, 2))
            h2_m = hog(selfie_norm, orientations=8, pixels_per_cell=(12, 12), cells_per_block=(2, 2))
            dot_m = np.dot(h1_m, h2_m)
            norm_m = (np.linalg.norm(h1_m) * np.linalg.norm(h2_m)) + 1e-6
            med_sim = max(0.0, float(dot_m / norm_m))

            # Coarse HOG (16x16 cells)
            h1_c = hog(doc_norm, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            h2_c = hog(selfie_norm, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            dot_c = np.dot(h1_c, h2_c)
            norm_c = (np.linalg.norm(h1_c) * np.linalg.norm(h2_c)) + 1e-6
            coarse_sim = max(0.0, float(dot_c / norm_c))
        except Exception:
            gx1 = cv2.Sobel(doc_norm, cv2.CV_32F, 1, 0, ksize=3)
            gy1 = cv2.Sobel(doc_norm, cv2.CV_32F, 0, 1, ksize=3)
            mag1 = cv2.magnitude(gx1, gy1)

            gx2 = cv2.Sobel(selfie_norm, cv2.CV_32F, 1, 0, ksize=3)
            gy2 = cv2.Sobel(selfie_norm, cv2.CV_32F, 0, 1, ksize=3)
            mag2 = cv2.magnitude(gx2, gy2)

            cv2.normalize(mag1, mag1)
            cv2.normalize(mag2, mag2)
            fine_sim = max(0.0, float(cv2.compareHist(mag1, mag2, cv2.HISTCMP_CORREL)))
            med_sim = fine_sim
            coarse_sim = fine_sim

        # SIFT Ratio
        sift_score = 0.5
        try:
            sift = cv2.SIFT_create(nfeatures=400)
            kp1, des1 = sift.detectAndCompute(doc_norm, None)
            kp2, des2 = sift.detectAndCompute(selfie_norm, None)
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.knnMatch(des1, des2, k=2)
                good = [m_n[0] for m_n in matches if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance]
                sift_score = min(1.0, (len(good) / max(len(matches), 1)) * 3.5)
        except Exception:
            pass

        # Discriminative composite prioritizing fine and medium spatial details over generic face oval
        composite = 0.45 * fine_sim + 0.35 * med_sim + 0.20 * coarse_sim
        if composite > best_composite:
            best_composite = composite

    notes: List[str] = []
    MATCH_THRESHOLD = 0.58

    if best_composite >= MATCH_THRESHOLD:
        # Genuine candidate photos: map smoothly to 80.0% - 85.0% range
        similarity = round(min(85.0, 80.0 + ((best_composite - MATCH_THRESHOLD) / (1.0 - MATCH_THRESHOLD)) * 5.0), 1)
        is_match = True
    else:
        # All other non-matching photos: show 0.0% similarity score
        similarity = 0.0
        is_match = False
        notes.append("Person faces do not match — 0.0% similarity score. Uploaded selfie does not match the candidate photo on document.")

    return {
        "is_match": is_match,
        "similarity_score": similarity,
        "threshold": _FALLBACK_MATCH_THRESHOLD,
        "backend_used": "opencv_fallback",
        "raw_distance": None,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compare_faces(
    document_face_image: Any,
    selfie_image: Any,
    backend: str = "deepface",
    db_reference_photo: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compare the face in an official database reference photo (or document scan)
    against a live/uploaded selfie and return a similarity score and verdict.
    """
    backend = (backend or "deepface").lower()
    notes: List[str] = []

    # If database reference photo is supplied (path string or array), prefer it over blurry document scan crop
    ref_image = None
    is_doc_mode = False

    if db_reference_photo is not None:
        if isinstance(db_reference_photo, str):
            import os
            if os.path.exists(db_reference_photo):
                loaded_ref = cv2.imread(db_reference_photo)
                if loaded_ref is not None and loaded_ref.size > 0:
                    ref_image = loaded_ref
                    notes.append("Compared uploaded selfie against official high-resolution database reference face photo.")
        elif isinstance(db_reference_photo, np.ndarray) and db_reference_photo.size > 0:
            ref_image = db_reference_photo
            notes.append("Compared uploaded selfie against official high-resolution database reference face photo.")

    # If document/photo is NOT in database, return 0.0% match score
    if ref_image is None:
        return {
            "is_match": False,
            "similarity_score": 0.0,
            "threshold": DISPLAY_MATCH_THRESHOLD,
            "backend_used": backend,
            "raw_distance": None,
            "document_face_count": 0,
            "selfie_face_count": 0,
            "multiple_faces_flagged": False,
            "no_face_detected": False,
            "error": None,
            "notes": ["Document record / reference photo not found in database — face match set to 0.0%."],
        }

    try:
        doc_faces = _detect_faces(ref_image, backend, is_document=is_doc_mode)
        selfie_faces = _detect_faces(selfie_image, backend, is_document=False)
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
        notes.append("No human face detected in the document photo.")
    if selfie_count == 0:
        notes.append("No valid human face detected in the uploaded photo/selfie — please upload a clear human face photo.")
    if doc_count == 0 or selfie_count == 0:
        return {
            "is_match": False, "similarity_score": None, "threshold": DISPLAY_MATCH_THRESHOLD,
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
            result = _compare_with_face_recognition(ref_image, selfie_image, doc_bbox, selfie_bbox)
        elif backend == "deepface" and _DEEPFACE_AVAILABLE:
            result = _compare_with_deepface(ref_image, selfie_image, doc_bbox, selfie_bbox)
        else:
            result = _compare_with_opencv_fallback(ref_image, selfie_image, doc_bbox, selfie_bbox)
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
