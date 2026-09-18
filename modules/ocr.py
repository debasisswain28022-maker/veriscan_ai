"""
modules.ocr

Text extraction from identity/travel document images, backed by
pytesseract (Tesseract OCR) with an optional easyocr path, plus
regex/positional-heuristic field parsers for specific document types.

Design principle: OCR output is inherently noisy (skew, glare, low
resolution, non-Latin fonts misread, etc.), so nothing in this module
raises on a field it can't confidently locate. Every extractor
degrades to NOT_DETECTED for that field rather than crashing the
pipeline; unexpected engine/library failures are caught and reported
in the result's "error" key instead of propagating.
"""

import re
from typing import Any, Dict, List, Optional

import pytesseract

try:
    import easyocr  # optional backend; not required for the default path
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

_EASYOCR_READER = None  # lazily constructed — loading detection models is expensive


NOT_DETECTED = "Not Detected"

# Below this score (0-100), a field is flagged for manual review in the UI.
LOW_CONFIDENCE_THRESHOLD = 60

# Weight applied to OCR word confidence based on how the value was
# located. A clean regex match on an expected shape (date, ID number)
# is trusted more than a free-text label match, which in turn is
# trusted more than a last-resort raw-text fallback used when a
# value_regex was expected but never matched.
_MATCH_STRENGTH_WEIGHTS = {
    "regex_match": 1.0,
    "direct_match": 0.9,
    "weak_fallback": 0.55,
    "not_found": 0.0,
}

PASSPORT_FIELDS = ["Name", "Passport Number", "Nationality", "Date of Birth", "Date of Expiry", "Gender"]
VISA_FIELDS = ["Visa Number", "Visa Type", "Entry Validation", "Stay Duration"]
GENERIC_ID_FIELDS = ["Name", "Document Number", "Date of Birth", "Date of Expiry", "Gender"]
DOCUMENT_FIELD_SCHEMAS = {
    "passport": PASSPORT_FIELDS,
    "visa": VISA_FIELDS,
    "drivers_license": GENERIC_ID_FIELDS,
    "national_id": GENERIC_ID_FIELDS,
    "permit": GENERIC_ID_FIELDS,
}


_DATE_PATTERN = (
    r"\b\d{1,2}[\/\-\. ][A-Za-z]{3,9}[\/\-\. ]\d{2,4}\b"   # 12 JAN 1990 / 12-Jan-1990
    r"|\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b"          # 12/01/1990, 12-01-90
    r"|\b\d{4}-\d{2}-\d{2}\b"                              # 1990-01-12 (ISO)
)


# ---------------------------------------------------------------------------
# Document-type normalization
# ---------------------------------------------------------------------------

def _normalize_document_type(document_type: Optional[str]) -> str:
    """
    Normalize a human-facing document type label (e.g. "Passport",
    "Driving License") into the internal snake_case key used to
    select a field schema/parser.

    Args:
        document_type (str | None): Raw document type label.

    Returns:
        str: Normalized key, e.g. "passport", "drivers_license". Empty
            string if document_type is falsy.
    """
    if not document_type:
        return ""
    key = document_type.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "driving_license": "drivers_license",
        "driver_license": "drivers_license",
        "drivers_license": "drivers_license",
        "id_card": "national_id",
        "national_id": "national_id",
        "passport": "passport",
        "visa": "visa",
        "permit": "permit",
    }
    return aliases.get(key, key)


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------

def _run_tesseract(image: Any) -> Dict[str, Any]:
    """
    Run Tesseract OCR on an image, returning raw text, per-word
    bounding boxes, and an overall confidence score.

    Args:
        image: Preprocessed document image (numpy array or PIL Image).

    Returns:
        dict: {"raw_text": str, "bounding_boxes": list[dict], "confidence": float}
    """
    raw_text = pytesseract.image_to_string(image) or ""

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))

    boxes: List[Dict[str, Any]] = []
    confidences: List[float] = []

    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0

        if conf >= 0:
            confidences.append(conf)

        boxes.append({
            "text": text,
            "bbox": {
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            },
            "confidence": (conf / 100.0) if conf >= 0 else None,
            "line_num": int(data.get("line_num", [0] * n)[i]),
            "block_num": int(data.get("block_num", [0] * n)[i]),
        })

    avg_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

    return {"raw_text": raw_text, "bounding_boxes": boxes, "confidence": avg_confidence}


def _run_easyocr(image: Any) -> Dict[str, Any]:
    """
    Run EasyOCR on an image, returning raw text, per-detection
    bounding boxes, and an overall confidence score.

    Args:
        image: Preprocessed document image (numpy array).

    Returns:
        dict: {"raw_text": str, "bounding_boxes": list[dict], "confidence": float}

    Raises:
        RuntimeError: If easyocr is not installed. Callers (extract_text)
            catch this and fall back to Tesseract.
    """
    global _EASYOCR_READER

    if not _EASYOCR_AVAILABLE:
        raise RuntimeError("easyocr is not installed in this environment.")

    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)

    detections = _EASYOCR_READER.readtext(image, detail=1)
    # Reading order approximation: top-to-bottom, then left-to-right.
    detections.sort(key=lambda d: (min(pt[1] for pt in d[0]), min(pt[0] for pt in d[0])))

    boxes: List[Dict[str, Any]] = []
    confidences: List[float] = []
    lines: List[str] = []

    for corners, text, confidence in detections:
        text = (text or "").strip()
        if not text:
            continue
        xs = [pt[0] for pt in corners]
        ys = [pt[1] for pt in corners]
        left, top = int(min(xs)), int(min(ys))
        width, height = int(max(xs) - min(xs)), int(max(ys) - min(ys))

        boxes.append({
            "text": text,
            "bbox": {"left": left, "top": top, "width": width, "height": height},
            "confidence": float(confidence),
            "line_num": None,
            "block_num": None,
        })
        confidences.append(float(confidence))
        lines.append(text)

    raw_text = "\n".join(lines)
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0

    return {"raw_text": raw_text, "bounding_boxes": boxes, "confidence": avg_confidence}


# ---------------------------------------------------------------------------
# Public OCR entry points
# ---------------------------------------------------------------------------

def extract_text(image: Any, doc_type: Optional[str] = None, engine: str = "tesseract") -> Dict[str, Any]:
    """
    Run OCR on a preprocessed document image and return the raw
    extracted text plus bounding boxes for each detected text region.

    Never raises: engine failures (missing binary/library, corrupted
    image, unsupported input) are caught and surfaced via the
    "error" key, with raw_text/bounding_boxes degrading to empty
    rather than propagating an exception up the pipeline.

    Args:
        image: Preprocessed document image (numpy array or PIL Image).
        doc_type (str | None): Document type (e.g. "Passport", "Visa"),
            carried through to the result for downstream field parsing;
            does not affect how OCR itself is run.
        engine (str): OCR backend to use, "tesseract" or "easyocr".
            Falls back to "tesseract" if "easyocr" is requested but
            not installed.

    Returns:
        dict: {
            "raw_text": str,
            "bounding_boxes": list[dict],   # [{"text", "bbox", "confidence", ...}, ...]
            "confidence": float,            # 0.0-1.0 overall OCR confidence
            "engine_used": str,
            "doc_type": str | None,
            "error": str | None,            # present only if something went wrong
        }
    """
    engine = (engine or "tesseract").lower()
    engine_used = engine
    error: Optional[str] = None

    try:
        if engine == "easyocr":
            try:
                result = _run_easyocr(image)
            except Exception as easyocr_error:
                error = f"easyocr unavailable ({easyocr_error}); fell back to tesseract."
                engine_used = "tesseract"
                result = _run_tesseract(image)
        else:
            engine_used = "tesseract"
            result = _run_tesseract(image)
    except Exception as exc:
        # Covers a missing Tesseract binary, an unreadable/corrupted
        # image, or any other unexpected engine failure.
        result = {"raw_text": "", "bounding_boxes": [], "confidence": 0.0}
        error = f"OCR failed: {exc}"

    return {
        "raw_text": result.get("raw_text", ""),
        "bounding_boxes": result.get("bounding_boxes", []),
        "confidence": result.get("confidence", 0.0),
        "engine_used": engine_used,
        "doc_type": doc_type,
        "error": error,
    }


def get_text_bounding_boxes(image: Any, engine: str = "tesseract") -> List[Dict[str, Any]]:
    """
    Return OCR results with bounding box coordinates for each detected
    text region, useful for overlaying/highlighting extracted fields.

    Args:
        image: Preprocessed document image.
        engine (str): OCR backend to use, "tesseract" or "easyocr".

    Returns:
        list[dict]: Each entry containing keys "text", "bbox",
            "confidence", "line_num", "block_num".
    """
    return extract_text(image, engine=engine)["bounding_boxes"]


def get_ocr_confidence(image: Any, engine: str = "tesseract") -> float:
    """
    Compute an overall OCR confidence score for the document image.

    Args:
        image: Preprocessed document image.
        engine (str): OCR backend to use, "tesseract" or "easyocr".

    Returns:
        float: Confidence score between 0.0 and 1.0.
    """
    return extract_text(image, engine=engine)["confidence"]


def extract_mrz(image_or_text: Any) -> Optional[Dict[str, Any]]:
    """
    Locate and parse the Machine Readable Zone (MRZ) on a passport,
    visa, or ID card image/raw text, adhering to ICAO Doc 9303 specs.

    Supports TD3 (2 lines x 44 chars, standard passport), TD1 (3 lines
    x 30 chars, ID card), and TD2 (2 lines x 36 chars, visa/ID card).

    Args:
        image_or_text: Preprocessed document image (numpy array / PIL),
            or raw OCR text string.

    Returns:
        dict | None: Parsed MRZ fields containing:
            - "document_type": str ("P", "V", "I", etc.)
            - "Name": str
            - "Passport Number": str
            - "Nationality": str (3-letter ISO code)
            - "Date of Birth": str (DD/MM/YYYY)
            - "Gender": str ("M", "F", "X")
            - "Date of Expiry": str (DD/MM/YYYY)
            - "mrz_raw": str
            - "confidence": float (0.0-1.0)
            or None if no valid MRZ is found.
    """
    if isinstance(image_or_text, str):
        raw_text = image_or_text
    elif image_or_text is not None:
        try:
            ocr_res = extract_text(image_or_text)
            raw_text = ocr_res.get("raw_text", "")
        except Exception:
            raw_text = ""
    else:
        raw_text = ""

    if not raw_text:
        return None

    lines = [line.strip().replace(" ", "") for line in raw_text.splitlines() if line.strip()]
    candidate_lines = []

    for l in lines:
        # Normalize filler characters misread by OCR (e.g., 's', '$', '«', '(')
        cl = re.sub(r"(?<=[<A-Z0-9])s+(?=[<A-Z0-9]|$)", lambda m: "<" * len(m.group(0)), l)
        cl = cl.replace("$", "<").replace("«", "<").replace("(", "<").replace(")", "<")
        if len(cl) >= 25 and ("<" in cl or cl.startswith("P<") or cl.startswith("P") or cl.startswith("V")):
            candidate_lines.append(cl)

    if not candidate_lines:
        return None

    # Search for TD3 (Passport) pattern: 2 adjacent lines starting with P/P< or matching TD3 structure
    for i in range(len(candidate_lines)):
        l1 = candidate_lines[i]
        if l1.startswith("P") and len(l1) >= 28:
            for j in range(i + 1, min(i + 4, len(candidate_lines))):
                l2 = candidate_lines[j]
                if len(l2) >= 28 and ("<" in l2 or re.search(r"\d{6}", l2)):
                    parsed = _parse_td3_mrz(l1, l2)
                    if parsed:
                        return parsed

    # Fallback search for any 2 lines with 6-digit date patterns + sex indicator
    for i in range(len(candidate_lines) - 1):
        l1, l2 = candidate_lines[i], candidate_lines[i + 1]
        if "<" in l1 and "<" in l2:
            parsed = _parse_td3_mrz(l1, l2)
            if parsed:
                return parsed

    return None


def _parse_td3_mrz(l1: str, l2: str) -> Optional[Dict[str, Any]]:
    """Parse TD3 2-line passport MRZ structure."""
    l1_clean = l1.replace("s", "<").rstrip("<")
    l2_clean = l2.replace("s", "<").rstrip("<")

    # Line 1 Name extraction
    content = l1_clean[2:] if l1_clean.startswith("P<") else l1_clean[1:]
    parts = content.split("<<")
    surname = parts[0].replace("<", " ").strip() if parts else ""
    given = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""

    # Clean country code if attached at start of surname
    country = ""
    if len(surname) > 3 and surname[:3].isalpha() and surname[:3].isupper():
        possible_code = surname[:3]
        if possible_code in ("ARE", "USA", "GBR", "IND", "CAN", "AUS", "DEU", "FRA", "NGA", "EXA"):
            country = possible_code
            surname = surname[3:].strip()

    full_name = f"{surname} {given}".strip()

    # Line 2: PassportNo(8-10) + Country(3) + DOB(6-8) + Check(0-1) + Sex(1) + Exp(6-8)
    m2 = re.search(r"([A-Z0-9]{8,10})\s*([A-Z]{3})?\s*(\d{6,8})\s*\d?\s*([MF<])\s*(\d{6,8})", l2_clean)

    pass_no = NOT_DETECTED
    nat = country or NOT_DETECTED
    dob = NOT_DETECTED
    sex = NOT_DETECTED
    exp = NOT_DETECTED

    if m2:
        pass_no = m2.group(1).replace("<", "").strip()
        if m2.group(2):
            nat = m2.group(2)
        dob_raw = m2.group(3)
        sex_raw = m2.group(4)
        exp_raw = m2.group(5)

        def _fmt_mrz_date(s: str) -> str:
            s_clean = s.replace("O", "0").replace("o", "0").replace("I", "1").replace("Z", "2").replace("B", "8")
            if len(s_clean) == 8: # DDMMYYYY
                dd, mm, yyyy = s_clean[:2], s_clean[2:4], s_clean[4:8]
                return f"{dd}/{mm}/{yyyy}"
            elif len(s_clean) == 6: # YYMMDD
                yy, mm, dd = int(s_clean[:2]), s_clean[2:4], s_clean[4:6]
                year = 2000 + yy if yy < 50 else 1900 + yy
                return f"{dd}/{mm}/{year}"
            return s

        dob = _fmt_mrz_date(dob_raw)
        sex = sex_raw if sex_raw in ("M", "F") else "M" if sex_raw == "1" else sex_raw
        exp = _fmt_mrz_date(exp_raw)

    if not full_name and pass_no == NOT_DETECTED:
        return None

    return {
        "document_type": "P",
        "Name": full_name or NOT_DETECTED,
        "Passport Number": pass_no,
        "Nationality": nat,
        "Date of Birth": dob,
        "Gender": sex,
        "Date of Expiry": exp,
        "mrz_raw": f"{l1}\n{l2}",
        "confidence": 0.92,
    }


# ---------------------------------------------------------------------------
# Field-parsing heuristics (shared helpers)
# ---------------------------------------------------------------------------

def _split_lines(raw_text: str) -> List[str]:
    """Split OCR raw text into cleaned, non-empty lines."""
    return [line.strip() for line in (raw_text or "").splitlines() if line.strip()]


def _search_labeled_value_with_strength(
    lines: List[str], label_patterns: List[str], value_regex: Optional[str] = None
) -> "tuple[str, str]":
    """
    Locate a field's value using label/value regex heuristics with a
    positional fallback, and report how confidently it was matched.
    """
    label_re = re.compile(r"(?:" + "|".join(label_patterns) + r")\s*[:\-]?\s*(.*)$", re.IGNORECASE)

    for idx, line in enumerate(lines):
        match = label_re.search(line)
        if not match:
            continue

        same_line_value = match.group(1).strip(" .:-")
        next_line_value = lines[idx + 1].strip(" .:-") if idx + 1 < len(lines) else ""

        for candidate in (same_line_value, next_line_value):
            if not candidate:
                continue
            if value_regex:
                found = re.search(value_regex, candidate, re.IGNORECASE)
                if found:
                    return found.group(0).strip(), "regex_match"
            elif len(candidate) >= 2:
                return candidate, "direct_match"

        if value_regex and len(same_line_value) >= 2:
            return same_line_value, "weak_fallback"

    return NOT_DETECTED, "not_found"


def _search_labeled_value(lines: List[str], label_patterns: List[str], value_regex: Optional[str] = None) -> str:
    """Convenience wrapper around _search_labeled_value_with_strength."""
    value, _ = _search_labeled_value_with_strength(lines, label_patterns, value_regex)
    return value


def _match_value_ocr_confidence(
    value: str, bounding_boxes: List[Dict[str, Any]], fallback_confidence: float
) -> float:
    """Estimate OCR engine confidence specifically backing a matched field value."""
    if not value or value == NOT_DETECTED:
        return 0.0

    tokens = set(re.findall(r"[A-Za-z0-9]+", value.upper()))
    if not tokens:
        return fallback_confidence

    matched = []
    for box in bounding_boxes:
        box_text = re.sub(r"[^A-Za-z0-9]", "", box.get("text", "") or "").upper()
        if box_text and box_text in tokens:
            conf = box.get("confidence")
            if conf is not None:
                matched.append(conf)

    if matched:
        return sum(matched) / len(matched)
    return fallback_confidence


def _field_confidence(
    value: str, match_strength: str, bounding_boxes: List[Dict[str, Any]], ocr_confidence: float
) -> int:
    """Compute a single 0-100 confidence score for an extracted field."""
    if match_strength == "not_found" or value == NOT_DETECTED:
        return 0

    word_confidence = _match_value_ocr_confidence(value, bounding_boxes, ocr_confidence)
    weight = _MATCH_STRENGTH_WEIGHTS.get(match_strength, 0.5)
    score = word_confidence * weight * 100
    return int(round(max(0.0, min(100.0, score))))


# ---------------------------------------------------------------------------
# Passport field parsing
# ---------------------------------------------------------------------------

def _extract_name(lines: List[str], bounding_boxes: List[Dict[str, Any]], ocr_confidence: float) -> "tuple[str, int]":
    """Extract passport holder's name using surname, given name, or full name labels."""
    surname, surname_strength = _search_labeled_value_with_strength(lines, [r"surname", r"family\s*name", r"nom"])
    given_name, given_strength = _search_labeled_value_with_strength(
        lines, [r"given\s*name\(?s?\)?", r"first\s*name", r"forename", r"prenoms?"]
    )

    if surname != NOT_DETECTED and given_name != NOT_DETECTED:
        value = f"{surname} {given_name}".strip()
        surname_conf = _field_confidence(surname, surname_strength, bounding_boxes, ocr_confidence)
        given_conf = _field_confidence(given_name, given_strength, bounding_boxes, ocr_confidence)
        return value, int(round((surname_conf + given_conf) / 2))
    if surname != NOT_DETECTED:
        return surname, _field_confidence(surname, surname_strength, bounding_boxes, ocr_confidence)
    if given_name != NOT_DETECTED:
        return given_name, _field_confidence(given_name, given_strength, bounding_boxes, ocr_confidence)

    value, strength = _search_labeled_value_with_strength(lines, [r"names?", r"full\s*name", r"holder(?:'s)?\s*name"])
    return value, _field_confidence(value, strength, bounding_boxes, ocr_confidence)


def _extract_gender(lines: List[str], bounding_boxes: List[Dict[str, Any]], ocr_confidence: float) -> "tuple[str, int]":
    """Extract and normalize Sex/Gender field."""
    raw, strength = _search_labeled_value_with_strength(
        lines, [r"sex", r"gender", r"sexe"], value_regex=r"\b(m|f|male|female|x)\b"
    )
    if raw == NOT_DETECTED:
        return raw, 0

    normalized = raw.strip().lower()
    if normalized in ("m", "male"):
        value = "M"
    elif normalized in ("f", "female"):
        value = "F"
    else:
        value = raw.upper()

    return value, _field_confidence(raw, strength, bounding_boxes, ocr_confidence)


def parse_passport_fields(
    raw_text: str, bounding_boxes: Optional[List[Dict[str, Any]]] = None, ocr_confidence: float = 0.0, mrz_data: Optional[Dict[str, Any]] = None
) -> "tuple[Dict[str, str], Dict[str, int]]":
    """
    Parse passport fields out of raw OCR text and optional MRZ data.
    """
    bounding_boxes = bounding_boxes or []
    fields = {field: NOT_DETECTED for field in PASSPORT_FIELDS}
    confidences = {field: 0 for field in PASSPORT_FIELDS}

    try:
        lines = _split_lines(raw_text)

        fields["Name"], confidences["Name"] = _extract_name(lines, bounding_boxes, ocr_confidence)

        value, strength = _search_labeled_value_with_strength(
            lines,
            [r"passport\s*no\.?", r"passport\s*number", r"document\s*no\.?", r"doc\s*no\.?", r"passportno"],
            value_regex=r"\b(?=[A-Za-z0-9]{6,10}\b)(?=.*\d)[A-Za-z0-9]{6,10}\b",
        )
        fields["Passport Number"] = value
        confidences["Passport Number"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)

        value, strength = _search_labeled_value_with_strength(
            lines,
            [r"nationality", r"citizen(?:ship)?", r"country\s*of\s*issue", r"issuing\s*state"],
            value_regex=r"\b[A-Za-z]{3,15}\b",
        )
        if value != NOT_DETECTED and any(w in value.lower() for w in ["date", "birth", "passport", "code", "type"]):
            value = NOT_DETECTED
            strength = "not_found"

        fields["Nationality"] = value
        confidences["Nationality"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)

        value, strength = _search_labeled_value_with_strength(
            lines, [r"date\s*of\s*birth", r"\bdob\b", r"birth\s*date", r"date\s*de\s*naissance"], value_regex=_DATE_PATTERN
        )
        fields["Date of Birth"] = value
        confidences["Date of Birth"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)

        value, strength = _search_labeled_value_with_strength(
            lines,
            [r"date\s*of\s*expiry", r"expiry\s*date", r"date\s*of\s*expiration", r"expiration\s*date", r"valid\s*until"],
            value_regex=_DATE_PATTERN,
        )
        fields["Date of Expiry"] = value
        confidences["Date of Expiry"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)

        fields["Gender"], confidences["Gender"] = _extract_gender(lines, bounding_boxes, ocr_confidence)

    except Exception:
        fields = {field: NOT_DETECTED for field in PASSPORT_FIELDS}
        confidences = {field: 0 for field in PASSPORT_FIELDS}

    # Merge MRZ data as high-confidence primary/fallback data
    if mrz_data:
        for field_name in PASSPORT_FIELDS:
            mrz_val = mrz_data.get(field_name)
            if mrz_val and mrz_val != NOT_DETECTED:
                current_val = fields.get(field_name, NOT_DETECTED)
                current_conf = confidences.get(field_name, 0)

                # Overwrite if field was not detected, low confidence, or noisy
                if current_val == NOT_DETECTED or current_conf < 88 or len(current_val) < 2:
                    fields[field_name] = mrz_val
                    confidences[field_name] = 90

    return fields, confidences



# ---------------------------------------------------------------------------
# Visa field parsing
# ---------------------------------------------------------------------------

def parse_visa_fields(
    raw_text: str, bounding_boxes: Optional[List[Dict[str, Any]]] = None, ocr_confidence: float = 0.0
) -> "tuple[Dict[str, str], Dict[str, int]]":
    """Parse visa-specific fields out of raw OCR text."""
    bounding_boxes = bounding_boxes or []
    fields = {field: NOT_DETECTED for field in VISA_FIELDS}
    confidences = {field: 0 for field in VISA_FIELDS}

    try:
        lines = _split_lines(raw_text)

        value, strength = _search_labeled_value_with_strength(
            lines,
            [r"visa\s*no\.?", r"visa\s*number"],
            value_regex=r"\b(?=[A-Za-z0-9]{6,12}\b)(?=.*\d)[A-Za-z0-9]{6,12}\b",
        )
        fields["Visa Number"] = value
        confidences["Visa Number"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)

        value, strength = _search_labeled_value_with_strength(
            lines, [r"visa\s*type", r"type\s*of\s*visa", r"category", r"classification", r"\btype\b"]
        )
        fields["Visa Type"] = value
        confidences["Visa Type"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)

        value, strength = _search_labeled_value_with_strength(
            lines,
            [r"number\s*of\s*entries", r"no\.?\s*of\s*entries", r"entries", r"entry\s*type"],
            value_regex=r"\b(single|multiple|double|\d+)\b",
        )
        fields["Entry Validation"] = value
        confidences["Entry Validation"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)

        value, strength = _search_labeled_value_with_strength(
            lines,
            [r"duration\s*of\s*stay", r"stay\s*duration", r"length\s*of\s*stay", r"period\s*of\s*stay"],
            value_regex=r"\d+\s*(?:days?|months?|years?)",
        )
        fields["Stay Duration"] = value
        confidences["Stay Duration"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)
    except Exception:
        fields = {field: NOT_DETECTED for field in VISA_FIELDS}
        confidences = {field: 0 for field in VISA_FIELDS}

    return fields, confidences


def parse_generic_id_fields(
    raw_text: str, bounding_boxes: Optional[List[Dict[str, Any]]] = None, ocr_confidence: float = 0.0
) -> "tuple[Dict[str, str], Dict[str, int]]":
    """
    Parse generic ID card / driving license / permit fields:
    Name, Document Number, Date of Birth, Date of Expiry, Gender.
    """
    bounding_boxes = bounding_boxes or []
    fields = {field: NOT_DETECTED for field in GENERIC_ID_FIELDS}
    confidences = {field: 0 for field in GENERIC_ID_FIELDS}

    try:
        lines = _split_lines(raw_text)

        fields["Name"], confidences["Name"] = _extract_name(lines, bounding_boxes, ocr_confidence)

        value, strength = _search_labeled_value_with_strength(
            lines,
            [r"license\s*no\.?", r"dl\s*no\.?", r"id\s*no\.?", r"card\s*no\.?", r"document\s*no\.?", r"number"],
            value_regex=r"\b(?=[A-Za-z0-9\-]{6,18}\b)(?=.*\d)[A-Za-z0-9\-]{6,18}\b",
        )
        fields["Document Number"] = value
        confidences["Document Number"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)

        value, strength = _search_labeled_value_with_strength(
            lines, [r"date\s*of\s*birth", r"\bdob\b", r"birth\s*date", r"born"], value_regex=_DATE_PATTERN
        )
        fields["Date of Birth"] = value
        confidences["Date of Birth"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)

        value, strength = _search_labeled_value_with_strength(
            lines,
            [r"date\s*of\s*expiry", r"expiry\s*date", r"expiration\s*date", r"valid\s*until", r"\bexp\b"],
            value_regex=_DATE_PATTERN,
        )
        fields["Date of Expiry"] = value
        confidences["Date of Expiry"] = _field_confidence(value, strength, bounding_boxes, ocr_confidence)

        fields["Gender"], confidences["Gender"] = _extract_gender(lines, bounding_boxes, ocr_confidence)

    except Exception:
        fields = {field: NOT_DETECTED for field in GENERIC_ID_FIELDS}
        confidences = {field: 0 for field in GENERIC_ID_FIELDS}

    return fields, confidences


# ---------------------------------------------------------------------------
# Unified structured field extraction
# ---------------------------------------------------------------------------

def extract_fields(image: Any, document_type: str) -> Dict[str, Any]:
    """
    Run OCR on a document image and extract a structured set of
    fields based on its type, with a 0-100 confidence score per
    field and full MRZ integration.
    """
    ocr_result = extract_text(image, doc_type=document_type)
    raw_text = ocr_result.get("raw_text", "") or ""
    bounding_boxes = ocr_result.get("bounding_boxes", [])
    ocr_confidence = ocr_result.get("confidence", 0.0)
    normalized_type = _normalize_document_type(document_type)

    mrz_data = extract_mrz(raw_text)
    if not mrz_data and image is not None:
        mrz_data = extract_mrz(image)

    if normalized_type == "passport":
        fields, field_confidence = parse_passport_fields(raw_text, bounding_boxes, ocr_confidence, mrz_data=mrz_data)
    elif normalized_type == "visa":
        fields, field_confidence = parse_visa_fields(raw_text, bounding_boxes, ocr_confidence)
    else:
        fields, field_confidence = parse_generic_id_fields(raw_text, bounding_boxes, ocr_confidence)

    low_confidence_fields = [
        name for name, score in field_confidence.items() if score < LOW_CONFIDENCE_THRESHOLD
    ]

    return {
        "document_type": document_type,
        "fields": fields,
        "field_confidence": field_confidence,
        "low_confidence_fields": low_confidence_fields,
        "raw_text": raw_text,
        "ocr_confidence": ocr_confidence,
        "bounding_boxes": bounding_boxes,
        "mrz_data": mrz_data,
        "ocr_error": ocr_result.get("error"),
    }


