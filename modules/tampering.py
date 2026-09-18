"""
modules.tampering

Image forensics and tampering/forgery detection for scanned or
photographed identity/travel documents.

Implements three detectors plus a metadata check, combined into a
single 0-100 tampering_score by detect_tampering():

    1. Photo replacement detection (detect_photo_substitution) —
       locates the document's photo region (via OpenCV's built-in
       face cascade) and compares its Error Level Analysis (ELA) and
       noise/sharpness signature against the rest of the page. A
       pasted-in photo is typically sourced from a different image
       and recompressed, so its compression/noise fingerprint tends
       to differ from the surrounding, uniformly-scanned document.

    2. Text manipulation detection (check_font_consistency) — uses
       connected-component/contour analysis to find individual glyphs,
       groups them into text lines, and checks baseline alignment and
       stroke-width consistency. A single altered character (e.g. one
       digit changed in a date of birth) often sits slightly off the
       shared baseline or has a subtly different stroke weight than
       its neighbors.

    3. Stamp forgery detection (check_security_features) — locates
       stamp-colored ink regions (red/blue, the most common official
       stamp colors) and checks their geometric regularity
       (circularity/shape) and ink-color consistency. NOTE: this
       project ships no library of genuine reference-stamp templates
       per issuing authority, so it uses geometric/color-consistency
       heuristics as a stand-in signal rather than true
       cv2.matchShapes() template matching against known-genuine
       stamps — a real deployment would substitute an authoritative
       template library here.

    4. Metadata analysis (analyze_metadata) — reads EXIF from the
       *original* uploaded file (a numpy array has already lost this
       data) looking for photo-editing software signatures or a
       capture/modification timestamp mismatch.

Design principle: mirrors the rest of the pipeline's "never crash"
contract. Every detector catches its own failures and degrades to a
non-suspicious, zero-confidence result with an explanatory note
rather than raising, so one failing check can never take down the
overall tampering assessment.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import ExifTags, Image

NOT_DETECTED = "Not Detected"

# Relative weight of each check in the combined tampering_score.
_CHECK_WEIGHTS = {
    "photo_substitution": 0.35,
    "text_manipulation": 0.30,
    "stamp_forgery": 0.20,
    "metadata": 0.15,
}

_RISK_LEVEL_THRESHOLDS = {"high": 60, "medium": 30}

_EDITING_SOFTWARE_SIGNATURES = [
    "photoshop", "gimp", "snapseed", "lightroom", "pixlr", "picsart",
    "canva", "affinity photo", "paint.net", "illustrator", "capture one",
    "photopea", "inkscape",
]

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


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Lazily load OpenCV's bundled frontal-face Haar cascade."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


# ---------------------------------------------------------------------------
# 1. Photo replacement detection
# ---------------------------------------------------------------------------

def error_level_analysis(image: Any, quality: int = 90) -> Dict[str, Any]:
    """
    Perform Error Level Analysis (ELA): recompress the image at a
    known JPEG quality and measure the pixel-wise difference from the
    input. A region recently pasted in from a different source tends
    to show a distinct error-level signature from the rest of an
    otherwise uniformly-compressed page.

    Args:
        image: Document image (or crop), grayscale or BGR.
        quality (int): JPEG quality (0-100) to recompress at.

    Returns:
        dict: {
            "ela_map": np.ndarray | None,   # grayscale error map, scaled to 0-255
            "mean_error": float,
            "std_error": float,
        }
    """
    if image is None or image.size == 0:
        return {"ela_map": None, "mean_error": 0.0, "std_error": 0.0}

    working = _to_bgr(image)
    success, encoded = cv2.imencode(".jpg", working, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return {"ela_map": None, "mean_error": 0.0, "std_error": 0.0}

    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(working, recompressed)
    ela_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float32)

    scale = 255.0 / (ela_gray.max() + 1e-6)
    ela_map_scaled = np.clip(ela_gray * scale, 0, 255).astype(np.uint8)

    return {"ela_map": ela_map_scaled, "mean_error": float(ela_gray.mean()), "std_error": float(ela_gray.std())}


def _locate_photo_region(image: np.ndarray) -> Optional[Dict[str, int]]:
    """
    Locate the document's photo region using face detection, padded
    out to approximate the full photo (not just the face crop).

    Args:
        image: BGR document image.

    Returns:
        dict | None: {"left", "top", "width", "height"} in pixels, or
            None if no face was detected.
    """
    try:
        gray = _to_gray(image)
        faces = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x, pad_y = int(w * 0.4), int(h * 0.6)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1 = min(image.shape[1], x + w + pad_x)
        y1 = min(image.shape[0], y + h + pad_y)
        return {"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0}
    except Exception:
        return None


def detect_photo_substitution(image: Any) -> Dict[str, Any]:
    """
    Detect signs of photo replacement: locate the document photo
    region and compare its ELA and noise/sharpness signature against
    the rest of the page.

    Never raises — degrades to a non-suspicious result with an
    explanatory note on any failure (including "no photo found",
    which is not itself suspicious).

    Args:
        image: Document image, ideally color (BGR) rather than the
            grayscale OCR-preprocessed image, since compression/noise
            detail is needed here.

    Returns:
        dict: {
            "is_suspicious": bool,
            "confidence": float,          # 0.0-1.0
            "photo_region_found": bool,
            "region_bbox": dict | None,
            "ela_ratio": float | None,     # photo ELA / background ELA
            "noise_ratio": float | None,   # photo noise / background noise
            "notes": list[str],
        }
    """
    try:
        if image is None:
            return {
                "is_suspicious": False, "confidence": 0.0, "photo_region_found": False,
                "region_bbox": None, "ela_ratio": None, "noise_ratio": None,
                "notes": ["No image provided."],
            }

        working = _to_bgr(image)
        region = _locate_photo_region(working)

        if region is None:
            return {
                "is_suspicious": False, "confidence": 0.0, "photo_region_found": False,
                "region_bbox": None, "ela_ratio": None, "noise_ratio": None,
                "notes": ["No face/photo region detected — photo substitution check skipped."],
            }

        x, y, w, h = region["left"], region["top"], region["width"], region["height"]
        photo_crop = working[y:y + h, x:x + w]

        mask = np.ones(working.shape[:2], dtype=np.uint8) * 255
        mask[y:y + h, x:x + w] = 0
        background = cv2.bitwise_and(working, working, mask=mask)

        ela_photo = error_level_analysis(photo_crop)
        ela_background = error_level_analysis(background)

        photo_noise = float(cv2.Laplacian(_to_gray(photo_crop), cv2.CV_64F).var())
        bg_noise = float(cv2.Laplacian(_to_gray(background), cv2.CV_64F).var())

        ela_ratio = (ela_photo["mean_error"] + 1e-6) / (ela_background["mean_error"] + 1e-6)
        noise_ratio = (photo_noise + 1e-6) / (bg_noise + 1e-6)

        ela_deviation = abs(float(np.log(ela_ratio)))
        noise_deviation = abs(float(np.log(noise_ratio)))

        # Heuristic normalization — tuned to catch clear outliers rather
        # than subtle ones, since these signals are inherently noisy.
        confidence = round(min(1.0, (ela_deviation / 2.0) * 0.6 + (noise_deviation / 3.0) * 0.4), 3)
        is_suspicious = confidence >= 0.5

        if is_suspicious:
            note = (
                f"Photo region's compression/noise signature differs notably from the rest "
                f"of the document (ELA ratio {ela_ratio:.2f}x, noise ratio {noise_ratio:.2f}x)."
            )
        else:
            note = "Photo region's compression/noise signature is broadly consistent with the rest of the document."

        return {
            "is_suspicious": is_suspicious,
            "confidence": confidence,
            "photo_region_found": True,
            "region_bbox": region,
            "ela_ratio": round(ela_ratio, 3),
            "noise_ratio": round(noise_ratio, 3),
            "notes": [note],
        }
    except Exception as exc:
        return {
            "is_suspicious": False, "confidence": 0.0, "photo_region_found": False,
            "region_bbox": None, "ela_ratio": None, "noise_ratio": None,
            "notes": [f"Photo substitution check failed: {exc}"],
        }


# ---------------------------------------------------------------------------
# 2. Text manipulation detection
# ---------------------------------------------------------------------------

def check_font_consistency(image: Any) -> Dict[str, Any]:
    """
    Detect text manipulation via contour/connected-component analysis:
    groups detected glyphs into text lines and checks (a) baseline
    alignment — how far each glyph's bottom edge sits from its line's
    shared baseline — and (b) stroke-width consistency within each
    line, since a single altered/pasted character often deviates
    slightly on one or both axes even when visually convincing.

    Never raises — degrades to a non-suspicious result if too few
    text regions are found to judge consistency, or on any error.

    Args:
        image: Document image, grayscale or BGR.

    Returns:
        dict: {
            "is_suspicious": bool,
            "confidence": float,             # 0.0-1.0
            "baseline_deviation": float,      # mean pixel deviation from line baselines
            "outlier_regions": list[dict],    # glyph bboxes flagged as stroke-width outliers
            "notes": list[str],
        }
    """
    try:
        if image is None:
            return {
                "is_suspicious": False, "confidence": 0.0, "baseline_deviation": 0.0,
                "outlier_regions": [], "notes": ["No image provided."],
            }

        gray = _to_gray(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Connect characters within a word/line (wide, short kernel) without
        # bridging separate text lines together.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)

        h_img, w_img = gray.shape[:2]
        min_area, max_area = 30, 0.05 * h_img * w_img

        glyphs = []
        for i in range(1, num_labels):  # skip background label 0
            x, y, w, h, area = stats[i]
            if area < min_area or area > max_area or w < 4 or h < 6:
                continue
            glyphs.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area),
                            "bottom": int(y + h), "cy": float(centroids[i][1])})

        if len(glyphs) < 6:
            return {
                "is_suspicious": False, "confidence": 0.0, "baseline_deviation": 0.0,
                "outlier_regions": [],
                "notes": ["Not enough distinct text regions detected to assess font/baseline consistency."],
            }

        glyphs.sort(key=lambda g: g["cy"])
        median_height = float(np.median([g["h"] for g in glyphs]))
        line_gap_threshold = max(8.0, median_height * 0.6)

        lines: List[List[Dict[str, Any]]] = [[glyphs[0]]]
        for g in glyphs[1:]:
            if abs(g["cy"] - lines[-1][-1]["cy"]) <= line_gap_threshold:
                lines[-1].append(g)
            else:
                lines.append([g])

        baseline_deviations: List[float] = []
        stroke_outliers: List[Dict[str, Any]] = []

        # Global stroke-width distribution, used as a fallback for lines
        # too sparse (<3 glyphs) to judge consistency on their own —
        # otherwise a lone anomalous glyph isolated into its own tiny
        # line cluster would never get compared against anything.
        all_ratios = np.array([g["area"] / max(g["h"], 1) for g in glyphs], dtype=np.float32)
        global_median_ratio, global_std_ratio = float(np.median(all_ratios)), float(np.std(all_ratios)) + 1e-6

        for line in lines:
            if len(line) < 3:
                for g in line:
                    ratio = g["area"] / max(g["h"], 1)
                    z_score = abs(ratio - global_median_ratio) / global_std_ratio
                    if z_score > 2.5:
                        stroke_outliers.append({
                            "bbox": {"left": g["x"], "top": g["y"], "width": g["w"], "height": g["h"]},
                            "z_score": round(float(z_score), 2),
                        })
                continue  # too few glyphs on this line to judge baseline consistency

            bottoms = np.array([g["bottom"] for g in line], dtype=np.float32)
            median_bottom = float(np.median(bottoms))
            baseline_deviations.extend(np.abs(bottoms - median_bottom).tolist())

            # Stroke-width proxy: ink area relative to glyph height.
            ratios = np.array([g["area"] / max(g["h"], 1) for g in line], dtype=np.float32)
            median_ratio, std_ratio = float(np.median(ratios)), float(np.std(ratios)) + 1e-6
            for g, ratio in zip(line, ratios):
                z_score = abs(ratio - median_ratio) / std_ratio
                if z_score > 2.5:
                    stroke_outliers.append({
                        "bbox": {"left": g["x"], "top": g["y"], "width": g["w"], "height": g["h"]},
                        "z_score": round(float(z_score), 2),
                    })

        mean_baseline_deviation = float(np.mean(baseline_deviations)) if baseline_deviations else 0.0
        baseline_confidence = min(1.0, mean_baseline_deviation / (median_height * 0.5 + 1e-6))
        outlier_confidence = min(1.0, len(stroke_outliers) / max(1, len(glyphs)) * 5)

        confidence = round(min(1.0, 0.6 * baseline_confidence + 0.4 * outlier_confidence), 3)
        is_suspicious = confidence >= 0.5

        if is_suspicious:
            note = (
                f"Detected {len(stroke_outliers)} character region(s) with stroke-width anomalies "
                f"and/or baseline misalignment averaging {mean_baseline_deviation:.1f}px."
            )
        else:
            note = "Text baseline alignment and stroke width are broadly consistent across the document."

        return {
            "is_suspicious": is_suspicious,
            "confidence": confidence,
            "baseline_deviation": round(mean_baseline_deviation, 2),
            "outlier_regions": stroke_outliers[:20],
            "notes": [note],
        }
    except Exception as exc:
        return {
            "is_suspicious": False, "confidence": 0.0, "baseline_deviation": 0.0,
            "outlier_regions": [], "notes": [f"Font/baseline consistency check failed: {exc}"],
        }


# ---------------------------------------------------------------------------
# 3. Stamp forgery detection
# ---------------------------------------------------------------------------

def check_security_features(image: Any, document_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Detect stamp/seal forgery signs: locate stamp-colored ink regions
    (red/blue — the most common official stamp colors) and assess
    their geometric regularity (circularity) and ink-color
    consistency as a proxy for genuineness.

    NOTE: this does not match against a library of real, genuine
    reference-stamp templates per issuing authority (none ship with
    this project) — it uses geometric/color heuristics as a stand-in
    signal. A production deployment should substitute true
    cv2.matchShapes() template matching against authoritative stamp
    references here.

    Never raises — degrades to a non-suspicious result if no
    stamp-colored regions are found (not itself suspicious — many
    documents have no visible ink stamp), or on any error.

    Args:
        image: Document image, ideally color (BGR).
        document_type (str | None): Reserved for future
            document-type-specific stamp expectations; not currently
            used to change the detection logic.

    Returns:
        dict: {
            "is_suspicious": bool,
            "confidence": float,               # 0.0-1.0
            "candidate_stamps": list[dict],     # all detected stamp-colored regions
            "notes": list[str],
        }
    """
    try:
        if image is None:
            return {"is_suspicious": False, "confidence": 0.0, "candidate_stamps": [], "notes": ["No image provided."]}

        working = _to_bgr(image)
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)

        # Broad ink-color ranges for typical official stamps.
        color_ranges = {
            "red": [((0, 60, 40), (10, 255, 255)), ((170, 60, 40), (180, 255, 255))],
            "blue": [((100, 60, 40), (130, 255, 255))],
        }

        h_img, w_img = working.shape[:2]
        img_area = h_img * w_img
        min_area, max_area = 0.003 * img_area, 0.20 * img_area

        candidates: List[Dict[str, Any]] = []
        for color_name, ranges in color_ranges.items():
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)

                x, y, w, h = cv2.boundingRect(contour)
                region_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)
                hue_values = hsv[:, :, 0][region_mask == 255]
                color_std = float(np.std(hue_values)) if hue_values.size else 0.0

                candidates.append({
                    "color": color_name,
                    "bbox": {"left": int(x), "top": int(y), "width": int(w), "height": int(h)},
                    "circularity": round(float(circularity), 3),
                    "color_std": round(color_std, 2),
                })

        if not candidates:
            return {
                "is_suspicious": False, "confidence": 0.0, "candidate_stamps": [],
                "notes": ["No stamp-colored regions detected — document may not carry a visible ink "
                          "stamp, or uses a non-color (embossed) seal."],
            }

        flagged = []
        for candidate in candidates:
            irregular_shape = candidate["circularity"] < 0.55
            inconsistent_color = candidate["color_std"] > 25.0
            if irregular_shape or inconsistent_color:
                flagged.append({**candidate, "irregular_shape": irregular_shape, "inconsistent_color": inconsistent_color})

        confidence = round(len(flagged) / len(candidates), 3) if flagged else 0.0
        is_suspicious = len(flagged) > 0

        if is_suspicious:
            note = f"{len(flagged)} of {len(candidates)} candidate stamp region(s) show irregular geometry or inconsistent ink color."
        else:
            note = f"Found {len(candidates)} stamp-like region(s); geometry and ink color appear consistent with a genuine stamp."

        return {"is_suspicious": is_suspicious, "confidence": confidence, "candidate_stamps": candidates, "notes": [note]}
    except Exception as exc:
        return {"is_suspicious": False, "confidence": 0.0, "candidate_stamps": [], "notes": [f"Stamp forgery check failed: {exc}"]}


# ---------------------------------------------------------------------------
# 4. Image metadata analysis
# ---------------------------------------------------------------------------

def analyze_metadata(file_path: Optional[str]) -> Dict[str, Any]:
    """
    Inspect an image file's EXIF metadata for signs of prior editing:
    a known photo-editing software signature, or a modification
    timestamp significantly later than the original capture time.

    Operates on the *original* uploaded file — a numpy array (as used
    elsewhere in the pipeline) has already lost EXIF data during
    decoding, so this needs the original file path.

    Never raises — degrades to a non-suspicious, low-confidence result
    on missing input, missing EXIF, or any read error (none of which
    are conclusive evidence of tampering on their own).

    Args:
        file_path (str | None): Path to the original uploaded image file.

    Returns:
        dict: {
            "has_exif": bool | None,
            "editing_software_detected": str | None,
            "date_inconsistency": bool,
            "is_suspicious": bool,
            "confidence": float,       # 0.0-1.0
            "notes": list[str],
        }
    """
    try:
        if not file_path:
            return {
                "has_exif": None, "editing_software_detected": None, "date_inconsistency": False,
                "is_suspicious": False, "confidence": 0.0,
                "notes": ["No original file path provided; metadata analysis skipped."],
            }

        with Image.open(file_path) as img:
            exif_raw = img.getexif()

        if not exif_raw or len(exif_raw) == 0:
            return {
                "has_exif": False, "editing_software_detected": None, "date_inconsistency": False,
                "is_suspicious": False, "confidence": 0.15,
                "notes": ["No EXIF metadata found. Common for scans/screenshots and not conclusive on "
                          "its own, but also consistent with metadata having been stripped after editing."],
            }

        exif = {ExifTags.TAGS.get(tag_id, tag_id): value for tag_id, value in exif_raw.items()}

        software = str(exif.get("Software", "") or "").strip()
        editing_software_detected = None
        if software:
            lowered = software.lower()
            for signature in _EDITING_SOFTWARE_SIGNATURES:
                if signature in lowered:
                    editing_software_detected = software
                    break

        date_inconsistency = False
        original_dt_str, modify_dt_str = exif.get("DateTimeOriginal"), exif.get("DateTime")
        if original_dt_str and modify_dt_str:
            try:
                fmt = "%Y:%m:%d %H:%M:%S"
                original_dt = datetime.strptime(str(original_dt_str), fmt)
                modify_dt = datetime.strptime(str(modify_dt_str), fmt)
                if (modify_dt - original_dt).total_seconds() > 3600:
                    date_inconsistency = True
            except ValueError:
                pass

        notes: List[str] = []
        confidence = 0.0
        if editing_software_detected:
            confidence += 0.7
            notes.append(f"EXIF 'Software' tag indicates editing with: {editing_software_detected}.")
        if date_inconsistency:
            confidence += 0.3
            notes.append("File's modification timestamp is significantly later than its original capture timestamp.")
        confidence = round(min(1.0, confidence), 3)

        if not notes:
            notes.append("EXIF metadata present with no editing-software signature or timestamp inconsistency detected.")

        return {
            "has_exif": True,
            "editing_software_detected": editing_software_detected,
            "date_inconsistency": date_inconsistency,
            "is_suspicious": confidence >= 0.5,
            "confidence": confidence,
            "notes": notes,
        }
    except Exception as exc:
        return {
            "has_exif": None, "editing_software_detected": None, "date_inconsistency": False,
            "is_suspicious": False, "confidence": 0.0, "notes": [f"Metadata analysis failed: {exc}"],
        }


# ---------------------------------------------------------------------------
# Not yet implemented (deferred)
# ---------------------------------------------------------------------------

def detect_copy_move_forgery(image: Any) -> List[Dict[str, Any]]:
    """
    Detect copy-move forgery (regions duplicated/pasted elsewhere in
    the same image) — e.g. keypoint matching (SIFT/ORB) against
    itself to find duplicated patches.

    Not yet implemented — deferred; not part of the three detectors
    requested for this module's initial build (photo substitution,
    text manipulation, stamp forgery).

    Args:
        image: Preprocessed document image.

    Returns:
        list[dict]: Detected duplicated regions with bounding boxes
            and similarity scores.
    """
    pass


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def detect_tampering(
    image: Any, document_type: Optional[str] = None, original_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run all tampering-detection checks on a document image and
    combine them into a single 0-100 tampering_score, with a
    breakdown of which individual checks fired.

    Never raises: each sub-check already degrades gracefully on its
    own; this function additionally guards every call so one failing
    detector can never take down the overall assessment.

    Args:
        image: Document image — ideally a color (BGR) image with only
            resizing applied (e.g. utils.preprocessing's "resized"
            stage), not the grayscale OCR-ready image, since
            color/compression detail matters for forensic analysis.
        document_type (str | None): e.g. "Passport", "Visa" — reserved
            for future document-type-specific checks.
        original_file_path (str | None): Path to the original,
            unprocessed uploaded file, used for EXIF metadata analysis
            (a numpy array has no EXIF data of its own).

    Returns:
        dict: {
            "tampering_score": int,      # 0-100, higher = more suspicious
            "risk_level": str,           # "low" | "medium" | "high"
            "flags": list[str],          # names of checks that fired ("is_suspicious")
            "details": {
                "photo_substitution": dict,
                "text_manipulation": dict,
                "stamp_forgery": dict,
                "metadata": dict,
            },
        }
    """
    try:
        photo_result = detect_photo_substitution(image)
    except Exception as exc:
        photo_result = {"is_suspicious": False, "confidence": 0.0, "notes": [f"Check failed: {exc}"]}

    try:
        text_result = check_font_consistency(image)
    except Exception as exc:
        text_result = {"is_suspicious": False, "confidence": 0.0, "notes": [f"Check failed: {exc}"]}

    try:
        stamp_result = check_security_features(image, document_type)
    except Exception as exc:
        stamp_result = {"is_suspicious": False, "confidence": 0.0, "notes": [f"Check failed: {exc}"]}

    try:
        metadata_result = analyze_metadata(original_file_path)
    except Exception as exc:
        metadata_result = {"is_suspicious": False, "confidence": 0.0, "notes": [f"Check failed: {exc}"]}

    details = {
        "photo_substitution": photo_result,
        "text_manipulation": text_result,
        "stamp_forgery": stamp_result,
        "metadata": metadata_result,
    }

    weighted_sum = sum(_CHECK_WEIGHTS[name] * result.get("confidence", 0.0) for name, result in details.items())
    tampering_score = int(round(max(0.0, min(1.0, weighted_sum)) * 100))

    if tampering_score >= _RISK_LEVEL_THRESHOLDS["high"]:
        risk_level = "high"
    elif tampering_score >= _RISK_LEVEL_THRESHOLDS["medium"]:
        risk_level = "medium"
    else:
        risk_level = "low"

    flags = [name for name, result in details.items() if result.get("is_suspicious")]

    return {"tampering_score": tampering_score, "risk_level": risk_level, "flags": flags, "details": details}
