"""
modules.database

Persistence layer for VeriScan AI, built on SQLite (the standard
library's sqlite3 module — no external dependency).

Implemented so far: the mock document registry ("documents" table)
used for the pipeline's "DB Check" step — a synthetic stand-in for a
real government/issuer document-status lookup (e.g. a passport
office, immigration database, or watchlist), seeded with fictional
records for this decision-support prototype. NONE of the data here
is real; document numbers, names, and nationalities are all made up.

Screening-record persistence (save/get/list/delete a completed
screening) is scaffolded below but not yet implemented — deferred to
a later prompt alongside modules.blockchain's audit ledger.
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from modules.ocr import NOT_DETECTED

DEFAULT_DB_PATH = "data/veriscan.db"

REGISTRY_STATUS_DISPLAY = {
    "valid": "Valid",
    "expired": "Expired",
    "blacklisted": "Blacklisted",
}

# Which OCR-extracted field holds the "document number" to look up,
# per document type. Mirrors modules.validation.DOCUMENT_NUMBER_FIELD;
# duplicated locally (rather than imported) so modules.database stays
# decoupled from modules.validation, matching the pattern already used
# between modules.ocr and modules.validation.
DOCUMENT_NUMBER_FIELD_BY_TYPE = {
    "passport": "Passport Number",
    "visa": "Visa Number",
    "national_id": "ID Number",
    "drivers_license": "License Number",
    "permit": "Permit Number",
}

# Synthetic seed data for the mock registry — entirely fictional names,
# nationalities, and document numbers for demonstration purposes only.
# (document_number, document_type, full_name, nationality, status, expiry_date, notes)
_SEED_DOCUMENTS = [
    ("X1234567", "passport", "JOHN MICHAEL DOE", "EXAMPLIAN", "valid", "2030-01-20", None),
    ("A7654321", "passport", "ALICE SMITH", "EXAMPLIAN", "valid", "2028-06-15", None),
    ("P9988776", "passport", "MARIA GARCIA", "SPANISH", "valid", "2029-11-02", None),
    ("Y2233445", "passport", "WEI CHEN", "CHINESE", "expired", "2019-03-10", None),
    ("Z8899001", "passport", "RAJ PATEL", "INDIAN", "expired", "2021-07-04", None),
    ("B1122334", "passport", "SAMUEL OKAFOR", "NIGERIAN", "blacklisted", "2031-05-01", "Reported stolen"),
    ("C5566778", "passport", "ELENA PETROVA", "RUSSIAN", "blacklisted", "2027-09-09", "Flagged in fraud investigation"),
    ("V9876543", "visa", "JOHN MICHAEL DOE", "EXAMPLIAN", "valid", "2026-12-31", None),
    ("V1112223", "visa", "FATIMA AL-SAYED", "EGYPTIAN", "expired", "2022-02-14", None),
    ("V4445556", "visa", "LUCAS MUELLER", "GERMAN", "blacklisted", "2025-08-20", "Overstay violation on record"),
    ("D6677889", "national_id", "SOPHIA ROSSI", "ITALIAN", "valid", None, None),
    ("D3334445", "drivers_license", "KAI TANAKA", "JAPANESE", "valid", None, None),
]


# ---------------------------------------------------------------------------
# Document-type normalization (mirrors modules.ocr / modules.validation)
# ---------------------------------------------------------------------------

def _normalize_document_type(document_type: Optional[str]) -> str:
    """Normalize a human-facing document type label to its internal key."""
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


def get_document_number_from_fields(fields: Dict[str, Any], document_type: str) -> str:
    """
    Pick out the document-number value from an OCR fields dict for a
    given document type, for use as input to lookup_document().

    Args:
        fields (dict): Extracted field name -> value pairs (e.g. from
            modules.ocr.extract_fields()["fields"]).
        document_type (str): e.g. "Passport", "Visa".

    Returns:
        str: The document number, or NOT_DETECTED if the field is
            missing or this document type has no number field mapped.
    """
    normalized = _normalize_document_type(document_type)
    field_name = DOCUMENT_NUMBER_FIELD_BY_TYPE.get(normalized)
    if not field_name or not isinstance(fields, dict):
        return NOT_DETECTED

    value = fields.get(field_name, NOT_DETECTED)
    if isinstance(value, dict):  # defensive: tolerate a future {"value": ...} shape
        value = value.get("value", NOT_DETECTED)
    return value if value else NOT_DETECTED


# ---------------------------------------------------------------------------
# Setup / seeding
# ---------------------------------------------------------------------------

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initialize the SQLite database: create the `documents` registry
    table if it doesn't already exist, and seed it with at least 10
    synthetic records on first run (only when the table is empty).
    Safe to call repeatedly — subsequent calls are no-ops beyond the
    cheap existence/count checks.

    Args:
        db_path (str): Filesystem path to the SQLite database file.
    """
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_number TEXT UNIQUE NOT NULL,
                document_type TEXT NOT NULL,
                full_name TEXT NOT NULL,
                nationality TEXT,
                status TEXT NOT NULL CHECK (status IN ('valid', 'expired', 'blacklisted')),
                expiry_date TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()

        existing_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if existing_count == 0:
            _seed_documents(conn)
    finally:
        conn.close()


def _seed_documents(conn: sqlite3.Connection) -> None:
    """Insert the synthetic seed registry records. Internal helper for init_db()."""
    created_at = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT OR IGNORE INTO documents
            (document_number, document_type, full_name, nationality, status, expiry_date, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [(*row, created_at) for row in _SEED_DOCUMENTS],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Registry lookup ("DB Check")
# ---------------------------------------------------------------------------

def lookup_document(doc_number: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Look up a document number in the mock registry and report its
    status: Valid, Expired, Blacklisted, or Not Found.

    Never raises: a missing/uninitialized database, connection error,
    or empty/undetected input all resolve to a "Not Found"-shaped
    response (with details in "error" where relevant) rather than an
    exception, so a DB Check failure never crashes the pipeline.

    Args:
        doc_number (str): Document/passport/visa number to look up.
        db_path (str): Filesystem path to the SQLite database file.

    Returns:
        dict: {
            "status": "Valid" | "Expired" | "Blacklisted" | "Not Found",
            "record": dict | None,  # full registry row if a match was found
            "error": str | None,
        }
    """
    if not doc_number or doc_number == NOT_DETECTED or not str(doc_number).strip():
        return {"status": "Not Found", "record": None, "error": None}

    normalized_number = str(doc_number).strip().upper()

    try:
        init_db(db_path)  # idempotent — ensures the table exists and is seeded
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_number = ?", (normalized_number,)
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return {"status": "Not Found", "record": None, "error": f"Registry lookup failed: {exc}"}

    if row is None:
        return {"status": "Not Found", "record": None, "error": None}

    record = dict(row)
    display_status = REGISTRY_STATUS_DISPLAY.get(record.get("status"), "Not Found")

    return {"status": display_status, "record": record, "error": None}


def list_documents(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    List every record in the mock registry — intended for an
    admin/demo view showing what the registry currently contains, not
    for the screening pipeline itself.

    Never raises: returns an empty list if the database can't be read.

    Args:
        db_path (str): Filesystem path to the SQLite database file.

    Returns:
        list[dict]: All registry rows, most recently added first.
    """
    try:
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM documents ORDER BY id DESC").fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Screening-record persistence (not yet implemented)
# ---------------------------------------------------------------------------

def save_screening_record(record: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> int:
    """
    Persist a completed screening result to the database.

    Not yet implemented — deferred to a later prompt alongside the
    audit ledger in modules.blockchain, so the two can be wired
    together (each screening record hash-chained into the ledger).

    Args:
        record (dict): Aggregated screening result (OCR fields,
            validation flags, tampering findings, face match score,
            risk assessment, timestamps, etc.).
        db_path (str): Filesystem path to the SQLite database file.

    Returns:
        int: The newly created record's ID.
    """
    pass


def get_screening_record(record_id: int, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single screening record by ID.

    Not yet implemented — see save_screening_record().

    Args:
        record_id (int): ID of the record to retrieve.
        db_path (str): Filesystem path to the SQLite database file.

    Returns:
        dict | None: The screening record, or None if not found.
    """
    pass


def list_screening_records(
    filters: Optional[Dict[str, Any]] = None, db_path: str = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """
    List screening records, optionally filtered (e.g. by risk level,
    date range, document type).

    Not yet implemented — see save_screening_record().

    Args:
        filters (dict | None): Optional filter criteria.
        db_path (str): Filesystem path to the SQLite database file.

    Returns:
        list[dict]: Matching screening records.
    """
    pass


def delete_screening_record(record_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Delete a screening record by ID.

    Not yet implemented — see save_screening_record().

    Args:
        record_id (int): ID of the record to delete.
        db_path (str): Filesystem path to the SQLite database file.

    Returns:
        bool: True if a record was deleted, False otherwise.
    """
    pass
