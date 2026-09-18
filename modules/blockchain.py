"""
modules.blockchain

Tamper-evident audit logging for screening records, implemented as a
local hash-chained ledger (a simplified blockchain): each entry links
to the previous entry's hash, so altering or removing any past entry
breaks the chain in a way verify_chain() can detect by re-hashing.

This does not require, and does not attempt to be, a public/distributed
blockchain — it's a local integrity mechanism, persisted as a JSON
array of entries at ledger_path.

Design principle: mirrors the rest of the pipeline's "never crash"
contract. A missing/corrupted ledger file degrades to an empty chain
(init_ledger recreates it) or a clearly reported integrity failure
(verify_chain), rather than raising and taking down the Audit Trail page.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

DEFAULT_LEDGER_PATH = "data/audit_ledger.json"

# SHA-256 hex digests are 64 characters; the genesis (first) entry
# links to a hash of all zeros since there is no real previous entry.
GENESIS_PREV_HASH = "0" * 64

# The exact fields hashed into each entry's own hash, in the order
# they're serialized — prev_hash is included, chaining this entry to
# everything before it. Changing any one of these fields on any past
# entry, or reordering/deleting entries, changes the hash and breaks
# the chain from that point forward.
_HASH_FIELDS = ["prev_hash", "timestamp", "document_id", "risk_score", "decision"]


# ---------------------------------------------------------------------------
# Ledger file I/O
# ---------------------------------------------------------------------------

def init_ledger(ledger_path: str = DEFAULT_LEDGER_PATH) -> None:
    """
    Initialize the audit ledger file if it does not already exist,
    creating parent directories as needed. Safe to call repeatedly —
    never overwrites an existing ledger.

    Args:
        ledger_path (str): Filesystem path to the ledger store.
    """
    directory = os.path.dirname(ledger_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(ledger_path):
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump([], f)


def _load_chain(ledger_path: str) -> List[Dict[str, Any]]:
    """
    Load the full chain from disk. A missing or corrupted file is
    treated as an empty chain rather than raising, so a fresh
    add_record() call can start clean and verify_chain() can report
    the corruption explicitly instead of crashing the caller.
    """
    if not os.path.exists(ledger_path):
        return []
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_chain(chain: List[Dict[str, Any]], ledger_path: str) -> None:
    """Persist the full chain to disk, creating parent directories as needed."""
    directory = os.path.dirname(ledger_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(chain, f, indent=2)


def list_ledger_entries(ledger_path: str = DEFAULT_LEDGER_PATH) -> List[Dict[str, Any]]:
    """
    Return every entry in the ledger, oldest first.

    Args:
        ledger_path (str): Filesystem path to the ledger store.

    Returns:
        list[dict]: All ledger entries (each with prev_hash, timestamp,
            document_id, risk_score, decision, hash, index).
    """
    return _load_chain(ledger_path)


def get_ledger_entry(record_id: int, ledger_path: str = DEFAULT_LEDGER_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single ledger entry by its chain index.

    Args:
        record_id (int): The entry's position in the chain (its "index" field).
        ledger_path (str): Filesystem path to the ledger store.

    Returns:
        dict | None: The ledger entry, or None if not found.
    """
    chain = _load_chain(ledger_path)
    for entry in chain:
        if entry.get("index") == record_id:
            return entry
    return None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def compute_record_hash(record: Dict[str, Any]) -> str:
    """
    Compute a deterministic SHA-256 hash of a record's chained fields
    (prev_hash, timestamp, document_id, risk_score, decision), so the
    same field values always produce the same hash regardless of key
    order in the input dict.

    Args:
        record (dict): Must contain at least prev_hash, timestamp,
            document_id, risk_score, and decision. Extra keys (e.g.
            "hash" or "index" on an already-built entry) are ignored.

    Returns:
        str: Hex-encoded SHA-256 digest.
    """
    canonical = {field: record.get(field) for field in _HASH_FIELDS}
    serialized = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def add_record(
    document_id: str,
    risk_score: float,
    decision: str,
    timestamp: Optional[str] = None,
    ledger_path: str = DEFAULT_LEDGER_PATH,
) -> Dict[str, Any]:
    """
    Append a new record to the hash-chained audit ledger, linking it
    to the current last entry's hash (or the genesis hash if the
    chain is empty).

    Args:
        document_id (str): Identifier for the screened document (e.g.
            the extracted passport/visa number, or "Not Detected").
        risk_score (float): The 0-100 risk score from modules.risk_engine.
        decision (str): The decision/verdict recorded for this scan
            (e.g. the risk band "HIGH", or a fuller decision string).
        timestamp (str | None): ISO 8601 timestamp; defaults to the
            current UTC time if not supplied.
        ledger_path (str): Filesystem path to the ledger store.

    Returns:
        dict: The newly created entry: {prev_hash, timestamp,
            document_id, risk_score, decision, hash, index}.
    """
    from utils.helpers import get_timestamp  # local import avoids a hard cross-package dependency at module load

    init_ledger(ledger_path)
    chain = _load_chain(ledger_path)

    prev_hash = chain[-1]["hash"] if chain else GENESIS_PREV_HASH
    record = {
        "prev_hash": prev_hash,
        "timestamp": timestamp or get_timestamp(),
        "document_id": document_id if document_id else "Not Detected",
        "risk_score": risk_score,
        "decision": decision,
    }
    entry = dict(record)
    entry["hash"] = compute_record_hash(record)
    entry["index"] = len(chain)

    chain.append(entry)
    _save_chain(chain, ledger_path)
    return entry


def append_to_ledger(record: Dict[str, Any], ledger_path: str = DEFAULT_LEDGER_PATH) -> Dict[str, Any]:
    """
    Backward-compatible wrapper around add_record(), kept for the
    module's original contract. Prefer add_record() for new code.

    Args:
        record (dict): Must contain "document_id", "risk_score", and
            "decision" (a "timestamp" key is honored if present).
        ledger_path (str): Filesystem path to the ledger store.

    Returns:
        dict: The newly created ledger entry.
    """
    return add_record(
        document_id=record.get("document_id"),
        risk_score=record.get("risk_score"),
        decision=record.get("decision"),
        timestamp=record.get("timestamp"),
        ledger_path=ledger_path,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_chain(ledger_path: str = DEFAULT_LEDGER_PATH) -> Dict[str, Any]:
    """
    Walk the ledger from the genesis entry forward, re-hashing each
    entry's chained fields and confirming both (a) the entry's stored
    hash matches its recomputed hash (its content hasn't been edited)
    and (b) its prev_hash matches the previous entry's actual hash
    (no entry has been inserted, removed, or reordered).

    Args:
        ledger_path (str): Filesystem path to the ledger store.

    Returns:
        dict: {
            "is_valid": bool,
            "total_records": int,
            "broken_at_index": int | None,  # first entry where a mismatch was found
            "reason": str | None,           # human-readable explanation if invalid
        }
    """
    chain = _load_chain(ledger_path)

    if not chain:
        return {"is_valid": True, "total_records": 0, "broken_at_index": None, "reason": None}

    expected_prev_hash = GENESIS_PREV_HASH
    for i, entry in enumerate(chain):
        if entry.get("prev_hash") != expected_prev_hash:
            return {
                "is_valid": False,
                "total_records": len(chain),
                "broken_at_index": i,
                "reason": f"Entry {i}'s prev_hash does not match entry {i - 1}'s actual hash "
                          f"(an entry may have been inserted, removed, or reordered).",
            }

        recomputed_hash = compute_record_hash(entry)
        if entry.get("hash") != recomputed_hash:
            return {
                "is_valid": False,
                "total_records": len(chain),
                "broken_at_index": i,
                "reason": f"Entry {i}'s stored hash does not match its recomputed hash "
                          f"(one of its fields was likely edited after the fact).",
            }

        expected_prev_hash = entry["hash"]

    return {"is_valid": True, "total_records": len(chain), "broken_at_index": None, "reason": None}


def verify_ledger_integrity(ledger_path: str = DEFAULT_LEDGER_PATH) -> Dict[str, Any]:
    """
    Backward-compatible wrapper around verify_chain(), kept for the
    module's original contract. Prefer verify_chain() for new code.

    Args:
        ledger_path (str): Filesystem path to the ledger store.

    Returns:
        dict: {"is_valid": bool, "broken_at_index": int | None}
    """
    result = verify_chain(ledger_path)
    return {"is_valid": result["is_valid"], "broken_at_index": result["broken_at_index"]}
