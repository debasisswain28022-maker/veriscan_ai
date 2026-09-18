"""
modules.validation

Field-level and cross-field validation of data extracted from
identity/travel documents: document-number format checks, date
validity/logical consistency, required-field presence, and
cross-field checks (e.g. visa entry validity vs. stay duration).

Design principle: mirrors modules.ocr's "never crash" contract. A
field that's missing, unparseable, or shaped unexpectedly becomes a
validation issue rather than an exception — the pipeline should
always be able to render *something* for a human reviewer.

Severity model:
    "warning" — the data couldn't be confirmed (missing field, unparseable
                date, format that doesn't match the expected shape). This
                is often just OCR noise, not proof of a bad document, so
                it calls for manual review rather than automatic failure.
    "error"   — a fact independently verifiable from the extracted data
                itself is logically wrong (expired document, birth date
                after expiry date, a stay duration that exceeds what an
                entry-validity type would typically allow). These drive
                an overall "fail" status.
"""

import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional

try:
    from dateutil import parser as _dateutil_parser
    _DATEUTIL_AVAILABLE = True
except ImportError:
    _DATEUTIL_AVAILABLE = False

from modules.ocr import NOT_DETECTED, PASSPORT_FIELDS, VISA_FIELDS


# ---------------------------------------------------------------------------
# Document-type / schema configuration
# ---------------------------------------------------------------------------

def _normalize_document_type(document_type: Optional[str]) -> str:
    """
    Normalize a human-facing document type label into the internal
    snake_case key used to select a schema/pattern. Mirrors
    modules.ocr._normalize_document_type so the two modules agree on
    document-type keys without importing each other's private helpers.

    Args:
        document_type (str | None): Raw document type label.

    Returns:
        str: Normalized key, e.g. "passport", "drivers_license".
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


REQUIRED_FIELDS = {
    "passport": PASSPORT_FIELDS,
    "visa": VISA_FIELDS,
}

DOCUMENT_NUMBER_FIELD = {
    "passport": "Passport Number",
    "visa": "Visa Number",
    "national_id": "ID Number",
    "drivers_license": "License Number",
    "permit": "Permit Number",
}

# Simplified, illustrative shape checks — real official formats vary by
# issuing authority and are not standardized across all countries. These
# are reasonable general-shape sanity checks, not government specs.
DOCUMENT_NUMBER_PATTERNS = {
    "passport": r"^[A-Z0-9]{6,9}$",
    "visa": r"^[A-Z0-9]{6,12}$",
    "national_id": r"^[A-Z0-9]{6,12}$",
    "drivers_license": r"^[A-Z0-9\-]{5,15}$",
    "permit": r"^[A-Z0-9\-]{4,15}$",
}

# Illustrative maximum reasonable stay (in days) per entry-validity type,
# used only for a cross-field sanity check — not an official regulatory
# table, since real limits vary by issuing country and visa category.
_MAX_STAY_DAYS_BY_ENTRY_TYPE = {
    "single": 180,
    "double": 270,
    "multiple": 365,
}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _issue(field: str, severity: str, message: str) -> Dict[str, str]:
    """Build a single validation issue entry."""
    return {"field": field, "severity": severity, "message": message}


def _field_value(fields: Dict[str, Any], field_name: str) -> str:
    """
    Read a plain string value out of a fields dict entry. Defensive
    against a future nested {"value": ...} shape as well as the
    current plain-string shape produced by modules.ocr.
    """
    raw = fields.get(field_name, NOT_DETECTED)
    if isinstance(raw, dict):
        return raw.get("value", NOT_DETECTED) or NOT_DETECTED
    return raw if raw else NOT_DETECTED


def _parse_date_flexible(date_str: str) -> Optional[date]:
    """
    Best-effort parsing of an OCR-extracted date string (which may be
    in any of several formats commonly seen on travel documents) into
    a date object.

    Args:
        date_str (str): Raw date string, e.g. "15 JAN 1990", "12/01/1990".

    Returns:
        date | None: Parsed date, or None if it couldn't be parsed.
    """
    if not date_str or date_str == NOT_DETECTED:
        return None

    cleaned = date_str.strip().strip(".")

    explicit_formats = [
        "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y",
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d-%m-%y",
        "%Y-%m-%d",
    ]
    for fmt in explicit_formats:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue

    if _DATEUTIL_AVAILABLE:
        try:
            # dayfirst=True: travel documents overwhelmingly present
            # dates as DD-MM(M)-YYYY rather than the US MM/DD/YYYY order.
            return _dateutil_parser.parse(cleaned, dayfirst=True, fuzzy=True).date()
        except (ValueError, OverflowError, TypeError):
            return None

    return None


def _parse_stay_duration_days(value: str) -> Optional[int]:
    """
    Extract an approximate number of days from a stay-duration string
    like "90 DAYS", "3 MONTHS", or "1 YEAR".

    Args:
        value (str): Raw stay-duration field value.

    Returns:
        int | None: Approximate duration in days, or None if unparseable.
    """
    if not value or value == NOT_DETECTED:
        return None
    match = re.search(r"(\d+)\s*(day|month|year)s?", value, re.IGNORECASE)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "day":
        return amount
    if unit == "month":
        return amount * 30
    if unit == "year":
        return amount * 365
    return None


# ---------------------------------------------------------------------------
# Individual checks (also usable standalone)
# ---------------------------------------------------------------------------

def validate_document_number(document_number: str, document_type: str) -> bool:
    """
    Validate the format of a document number for the given document
    type against a simplified, illustrative shape pattern.

    Args:
        document_number (str): Extracted document number.
        document_type (str): e.g. "Passport", "Visa", "National ID".

    Returns:
        bool: True if the document number matches the expected shape.
    """
    normalized = _normalize_document_type(document_type)
    pattern = DOCUMENT_NUMBER_PATTERNS.get(normalized)
    if not pattern or not document_number or document_number == NOT_DETECTED:
        return False
    return bool(re.match(pattern, document_number.strip().upper()))


def check_expiry(expiry_date: str) -> Dict[str, Any]:
    """
    Check whether a document's expiry date has passed or is
    approaching.

    Args:
        expiry_date (str): Extracted expiry date string.

    Returns:
        dict: {"is_expired": bool | None, "days_until_expiry": int | None}
            None values indicate the date could not be parsed.
    """
    parsed = _parse_date_flexible(expiry_date)
    if parsed is None:
        return {"is_expired": None, "days_until_expiry": None}
    days_remaining = (parsed - date.today()).days
    return {"is_expired": days_remaining < 0, "days_until_expiry": days_remaining}


def validate_mrz_checksum(mrz_fields: Dict[str, str]) -> Dict[str, bool]:
    """
    Verify MRZ checksum digits (document number, DOB, expiry,
    composite) against the ICAO 9303 checksum algorithm.

    Not yet implemented — deferred until modules.ocr.extract_mrz is
    implemented, since there's no MRZ data to validate yet.

    Args:
        mrz_fields (dict): Parsed MRZ fields including checksum digits.

    Returns:
        dict: Per-field checksum pass/fail results.
    """
    pass


def cross_check_fields(visual_fields: Dict[str, str], mrz_fields: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Compare fields extracted from the visual inspection zone against
    the MRZ (or barcode) data to detect mismatches.

    Not yet implemented — deferred until modules.ocr.extract_mrz is
    implemented, since there's no MRZ data to cross-check yet.

    Args:
        visual_fields (dict): Fields parsed from the visible document text.
        mrz_fields (dict): Fields parsed from the MRZ/barcode.

    Returns:
        list[dict]: List of discrepancies found, each with field name,
            visual value, and MRZ value.
    """
    pass


# ---------------------------------------------------------------------------
# Check groups used by validate_document
# ---------------------------------------------------------------------------

def check_required_fields(fields: Dict[str, Any], document_type: str) -> List[Dict[str, str]]:
    """
    Check that every field in the document type's schema was
    successfully extracted (non-empty, not NOT_DETECTED).

    Args:
        fields (dict): Extracted field name -> value pairs.
        document_type (str): e.g. "Passport", "Visa".

    Returns:
        list[dict]: One "warning" issue per missing required field.
    """
    normalized = _normalize_document_type(document_type)
    required = REQUIRED_FIELDS.get(normalized, [])
    issues = []
    for field_name in required:
        value = _field_value(fields, field_name)
        if not value or value == NOT_DETECTED or not str(value).strip():
            issues.append(_issue(
                field_name, "warning",
                f"'{field_name}' could not be extracted and requires manual review.",
            ))
    return issues


def check_document_number_format(fields: Dict[str, Any], document_type: str) -> List[Dict[str, str]]:
    """
    Check the document's ID/passport/visa number against its expected
    format for the given document type.

    Args:
        fields (dict): Extracted field name -> value pairs.
        document_type (str): e.g. "Passport", "Visa".

    Returns:
        list[dict]: A "warning" issue if the number was extracted but
            doesn't match the expected shape. Empty if the field is
            missing (already covered by check_required_fields) or the
            format is fine.
    """
    normalized = _normalize_document_type(document_type)
    field_name = DOCUMENT_NUMBER_FIELD.get(normalized)
    if not field_name:
        return []

    value = _field_value(fields, field_name)
    if not value or value == NOT_DETECTED:
        return []

    if not validate_document_number(value, document_type):
        return [_issue(
            field_name, "warning",
            f"'{value}' does not match the expected format for {document_type} numbers.",
        )]
    return []


def check_date_logic(fields: Dict[str, Any], document_type: str) -> List[Dict[str, str]]:
    """
    Check that Date of Birth and Date of Expiry (where applicable for
    the document type) are valid, parseable dates that are logically
    consistent: DOB not in the future, expiry after DOB, and flag
    (rather than silently ignore) an already-expired document.

    Args:
        fields (dict): Extracted field name -> value pairs.
        document_type (str): e.g. "Passport".

    Returns:
        list[dict]: Issues found — "warning" for unparseable dates,
            "error" for logically inconsistent or expired dates.
    """
    normalized = _normalize_document_type(document_type)
    issues: List[Dict[str, str]] = []

    if normalized != "passport":
        return issues

    dob_raw = _field_value(fields, "Date of Birth")
    expiry_raw = _field_value(fields, "Date of Expiry")
    dob = _parse_date_flexible(dob_raw)
    expiry = _parse_date_flexible(expiry_raw)
    today = date.today()

    if dob_raw != NOT_DETECTED and dob is None:
        issues.append(_issue("Date of Birth", "warning", f"'{dob_raw}' could not be parsed as a valid date."))
    if expiry_raw != NOT_DETECTED and expiry is None:
        issues.append(_issue("Date of Expiry", "warning", f"'{expiry_raw}' could not be parsed as a valid date."))

    if dob is not None and dob > today:
        issues.append(_issue("Date of Birth", "error", f"Date of birth '{dob_raw}' is in the future."))

    if expiry is not None and expiry < today:
        issues.append(_issue("Date of Expiry", "error", f"Document expired on {expiry.isoformat()}."))

    if dob is not None and expiry is not None and dob >= expiry:
        issues.append(_issue("Date of Expiry", "error", "Date of expiry is not after date of birth."))

    return issues


def check_cross_field_logic(fields: Dict[str, Any], document_type: str) -> List[Dict[str, str]]:
    """
    Check cross-field logical consistency between related fields on
    the same document — currently: a visa's requested stay duration
    against what its entry-validity type (single/double/multiple)
    would typically allow.

    Args:
        fields (dict): Extracted field name -> value pairs.
        document_type (str): e.g. "Visa".

    Returns:
        list[dict]: An "error" issue if the stay duration exceeds the
            typical maximum for the visa's entry-validity type.
            Empty if either field is missing/unparseable or the
            combination is within reason.
    """
    normalized = _normalize_document_type(document_type)
    issues: List[Dict[str, str]] = []

    if normalized != "visa":
        return issues

    entry_raw = _field_value(fields, "Entry Validation")
    stay_raw = _field_value(fields, "Stay Duration")

    if entry_raw == NOT_DETECTED or stay_raw == NOT_DETECTED:
        return issues

    entry_key = entry_raw.strip().lower()
    max_days = _MAX_STAY_DAYS_BY_ENTRY_TYPE.get(entry_key)
    if max_days is None and entry_key.isdigit():
        entry_count = int(entry_key)
        if entry_count <= 1:
            max_days = _MAX_STAY_DAYS_BY_ENTRY_TYPE["single"]
        elif entry_count == 2:
            max_days = _MAX_STAY_DAYS_BY_ENTRY_TYPE["double"]
        else:
            max_days = _MAX_STAY_DAYS_BY_ENTRY_TYPE["multiple"]

    stay_days = _parse_stay_duration_days(stay_raw)

    if max_days is not None and stay_days is not None and stay_days > max_days:
        issues.append(_issue(
            "Stay Duration", "error",
            f"Requested stay of {stay_days} days exceeds the typical maximum of "
            f"{max_days} days for a '{entry_raw}' entry visa.",
        ))

    return issues


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def validate_document(fields: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
    """
    Run the full validation suite on a set of OCR-extracted document
    fields: required-field presence, document-number format, date
    validity/logical consistency, and cross-field checks.

    Never raises: an unexpected field shape or internal error becomes
    a "warning" issue rather than an exception, so the pipeline always
    has something to show a reviewer.

    Args:
        fields (dict): Extracted field name -> value pairs, as
            returned by modules.ocr.extract_fields()["fields"]. For
            document types without an OCR field parser yet, this may
            instead be a single {"note": ...} dict.
        doc_type (str): e.g. "Passport", "Visa", "National ID",
            "Driving License", "Permit".

    Returns:
        dict: {
            "document_type": str,
            "status": "pass" | "warning" | "fail",
            "issues": list[dict],   # [{"field", "severity", "message"}, ...]
        }
    """
    issues: List[Dict[str, str]] = []

    try:
        if isinstance(fields, dict) and set(fields.keys()) == {"note"}:
            issues.append(_issue("_document_type", "warning", fields["note"]))
        else:
            issues.extend(check_required_fields(fields, doc_type))
            issues.extend(check_document_number_format(fields, doc_type))
            issues.extend(check_date_logic(fields, doc_type))
            issues.extend(check_cross_field_logic(fields, doc_type))
    except Exception as exc:
        issues = [_issue("_general", "warning", f"Validation encountered an unexpected error: {exc}")]

    if any(issue["severity"] == "error" for issue in issues):
        status = "fail"
    elif issues:
        status = "warning"
    else:
        status = "pass"

    return {"document_type": doc_type, "status": status, "issues": issues}


def validate_fields(fields: Dict[str, str], document_type: str) -> Dict[str, Any]:
    """
    Backward-compatible wrapper around validate_document(), kept for
    the module's original contract. Prefer validate_document() for new
    code — it returns richer per-issue detail (field, severity,
    message) and a three-way pass/warning/fail status.

    Args:
        fields (dict): Extracted field name -> value pairs.
        document_type (str): e.g. "passport", "drivers_license", "national_id".

    Returns:
        dict: {"is_valid": bool, "field_errors": {field: message}, "warnings": [message, ...]}
    """
    result = validate_document(fields, document_type)
    field_errors = {issue["field"]: issue["message"] for issue in result["issues"] if issue["severity"] == "error"}
    warnings = [issue["message"] for issue in result["issues"] if issue["severity"] == "warning"]
    return {"is_valid": result["status"] != "fail", "field_errors": field_errors, "warnings": warnings}
