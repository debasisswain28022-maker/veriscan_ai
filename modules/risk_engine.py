"""
modules.risk_engine

Aggregates signals from document validation, the mock registry check,
tampering detection, and face verification into a single 0-100 risk
score with a decision-support band and a human-readable explanation.

This module produces a recommendation for human review, not an
automated accept/reject decision — the final call always belongs to a
person.

Design principle: mirrors the rest of the pipeline's "never crash"
contract. A missing or unparseable signal (e.g. no selfie was
supplied, so there's no face-match score) is excluded from scoring
and its weight is redistributed among the signals that *are*
available, rather than being treated as either "safe" or "maximally
risky" by default — both of those would misrepresent an absence of
data as a finding.
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configurable weights — tune freely; they're renormalized at call time
# over whichever signals are actually available, so they don't need to
# sum to 1.0 here.
# ---------------------------------------------------------------------------
RISK_WEIGHTS: Dict[str, float] = {
    "validation": 0.25,
    "db_status": 0.30,
    "tampering": 0.30,
    "face_match": 0.15,
}

# Risk bands over the final 0-100 score.
RISK_BANDS = [
    ("LOW", 0, 20),
    ("MEDIUM", 21, 50),
    ("HIGH", 51, 75),
    ("CRITICAL", 76, 100),
]

# Illustrative risk contribution (0-100, higher = riskier) per registry
# status returned by modules.database.lookup_document.
_DB_STATUS_RISK = {
    "valid": 0,
    "not found": 35,
    "expired": 65,
    "blacklisted": 100,
    "error": 50,  # the lookup itself failed — unknown, treated as medium risk
}

_FACTOR_LABELS = {
    "validation": "Document validation",
    "db_status": "Registry check",
    "tampering": "Tampering analysis",
    "face_match": "Face match",
}

# Hard floors: some signals are definitive, binary red flags rather than
# fuzzy heuristics, and shouldn't be diluted by an average with unrelated
# clean signals. A registry blacklist hit floors the score into the
# CRITICAL band regardless of how clean everything else looks — a human
# reviewer should never see "blacklisted" quietly averaged down to MEDIUM.
_DB_STATUS_SCORE_FLOOR = {"blacklisted": 80}


# ---------------------------------------------------------------------------
# Per-signal risk sub-scores
# ---------------------------------------------------------------------------

def _validation_risk(validation_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convert a modules.validation.validate_document() result into a
    0-100 risk contribution plus a short human-readable detail.

    Args:
        validation_result (dict | None): {"status", "issues", ...}.

    Returns:
        dict | None: {"score": float, "detail": str}, or None if no
            validation result was supplied.
    """
    if not validation_result:
        return None

    status = validation_result.get("status", "warning")
    issues = validation_result.get("issues", []) or []
    num_errors = sum(1 for issue in issues if issue.get("severity") == "error")
    num_warnings = sum(1 for issue in issues if issue.get("severity") == "warning")

    if status == "pass":
        return {"score": 0.0, "detail": "All fields validated with no issues."}
    if status == "warning":
        score = min(100.0, 20.0 + 10.0 * num_warnings)
        return {"score": score, "detail": f"{num_warnings} field(s) flagged for manual review."}
    if status == "fail":
        score = min(100.0, 60.0 + 15.0 * num_errors)
        return {"score": score, "detail": f"{num_errors} validation error(s), e.g. expired or logically inconsistent dates."}

    return {"score": 50.0, "detail": "Validation status could not be determined."}


def _db_status_risk(db_status: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Convert a modules.database.lookup_document() status string into a
    0-100 risk contribution plus a short human-readable detail.

    Args:
        db_status (str | None): e.g. "Valid", "Expired", "Blacklisted", "Not Found".

    Returns:
        dict | None: {"score": float, "detail": str}, or None if no
            status was supplied.
    """
    if not db_status:
        return None

    key = db_status.strip().lower()
    score = _DB_STATUS_RISK.get(key, 50.0)

    details = {
        "valid": "Document number found in the registry with a valid status.",
        "not found": "Document number was not found in the registry.",
        "expired": "Document number is registered but marked expired.",
        "blacklisted": "Document number is registered as BLACKLISTED.",
        "error": "Registry lookup could not be completed.",
    }
    return {"score": float(score), "detail": details.get(key, f"Unrecognized registry status '{db_status}'.")}


def _tampering_risk(tampering_score: Optional[float]) -> Optional[Dict[str, Any]]:
    """
    Convert a modules.tampering.detect_tampering() tampering_score
    (already 0-100, higher = more suspicious) into a risk contribution.

    Args:
        tampering_score (float | None): 0-100 tampering score.

    Returns:
        dict | None: {"score": float, "detail": str}, or None if no
            tampering score was supplied.
    """
    if tampering_score is None:
        return None

    score = max(0.0, min(100.0, float(tampering_score)))
    if score < 30:
        detail = "No significant signs of tampering detected."
    elif score < 60:
        detail = "Some tampering indicators present; recommend closer inspection."
    else:
        detail = "Multiple strong tampering indicators detected."
    return {"score": score, "detail": detail}


def _face_match_risk(face_match_score: Optional[float]) -> Optional[Dict[str, Any]]:
    """
    Convert a modules.face_verification.compare_faces() similarity
    score (0-100, higher = more similar) into a risk contribution
    (inverted: low similarity = high risk).

    Args:
        face_match_score (float | None): 0-100 similarity score, or
            None if no selfie was supplied / no verdict was reached.

    Returns:
        dict | None: {"score": float, "detail": str}, or None if no
            usable face-match score was supplied.
    """
    if face_match_score is None:
        return None

    similarity = max(0.0, min(100.0, float(face_match_score)))
    risk = 100.0 - similarity
    if similarity >= 70:
        detail = f"Selfie closely matches the document photo ({similarity:.0f}% similarity)."
    elif similarity >= 50:
        detail = f"Selfie similarity to the document photo is borderline ({similarity:.0f}%)."
    else:
        detail = f"Selfie does not appear to match the document photo ({similarity:.0f}% similarity)."
    return {"score": risk, "detail": detail}


# ---------------------------------------------------------------------------
# Classification & explanation
# ---------------------------------------------------------------------------

def classify_risk_level(risk_score: float) -> str:
    """
    Map a numeric 0-100 risk score to one of the four risk bands.

    Args:
        risk_score (float): Aggregated risk score, 0-100.

    Returns:
        str: "LOW" (0-20), "MEDIUM" (21-50), "HIGH" (51-75), or
            "CRITICAL" (76-100).
    """
    clamped = max(0.0, min(100.0, risk_score))
    for band_name, low, high in RISK_BANDS:
        if low <= clamped <= high:
            return band_name
    return "CRITICAL"  # defensive fallback; unreachable given the bands above


_BAND_HEADLINES = {
    "LOW": "Low risk — no significant concerns identified.",
    "MEDIUM": "Medium risk — recommend manual review before proceeding.",
    "HIGH": "High risk — recommend escalation and thorough manual review.",
    "CRITICAL": "Critical risk — recommend rejecting or escalating for investigation.",
}


def generate_recommendation(risk_level: str, contributing_factors: List[Dict[str, Any]]) -> str:
    """
    Produce a human-readable decision-support recommendation for a
    reviewer, based on the risk band and the factors that contributed
    most to the score.

    Args:
        risk_level (str): Output of classify_risk_level() — "LOW",
            "MEDIUM", "HIGH", or "CRITICAL".
        contributing_factors (list[dict]): Factors as built by
            compute_risk_score(), each with "factor", "detail",
            "weighted_score", sorted by weighted_score descending.

    Returns:
        str: A short explanation paragraph, e.g. "Medium risk —
            recommend manual review before proceeding. Primary
            contributors: Registry check — document number is
            registered but marked expired; ..."
    """
    headline = _BAND_HEADLINES.get(risk_level, _BAND_HEADLINES["MEDIUM"])

    # Only call out factors that meaningfully moved the score — a
    # near-zero contributor isn't worth listing as a "driver".
    notable = [f for f in contributing_factors if f.get("weighted_score", 0) >= 2.0][:3]
    if not notable:
        return headline

    factor_sentences = [f"{f['factor']} — {f['detail']}" for f in notable]
    return f"{headline} Primary contributors: {'; '.join(factor_sentences)}."


def get_scoring_weights() -> Dict[str, float]:
    """
    Return the configurable weights used to combine individual
    signals (validation, registry status, tampering, face match) into
    the overall risk score. Edit RISK_WEIGHTS at the top of this file
    to tune them — they're renormalized at call time over whichever
    signals are actually available, so they don't need to sum to 1.0.

    Returns:
        dict: Signal name -> weight.
    """
    return dict(RISK_WEIGHTS)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def compute_risk_score(
    validation_result: Optional[Dict[str, Any]],
    db_status: Optional[str],
    tampering_score: Optional[float],
    face_match_score: Optional[float],
) -> Dict[str, Any]:
    """
    Combine document validation, registry status, tampering score, and
    face-match score into a single weighted 0-100 risk score, banded
    into LOW/MEDIUM/HIGH/CRITICAL, with a breakdown of which factors
    contributed most.

    Any signal that's unavailable (None) — most commonly face_match_score
    when no selfie was supplied — is excluded from scoring rather than
    scored as either safe or risky, and RISK_WEIGHTS is renormalized
    over whichever signals remain so their relative weighting is
    preserved.

    A registry blacklist hit is treated as a hard floor (see
    _DB_STATUS_SCORE_FLOOR) rather than just another weighted input —
    it pushes the score into the CRITICAL band regardless of how clean
    every other signal looks, since diluting a definitive blacklist hit
    with an average could mislead a reviewer into deprioritizing it.

    Never raises: a malformed input degrades that signal to "excluded"
    rather than propagating an exception.

    Args:
        validation_result (dict | None): Output of
            modules.validation.validate_document().
        db_status (str | None): Output of
            modules.database.lookup_document()["status"].
        tampering_score (float | None): Output of
            modules.tampering.detect_tampering()["tampering_score"] (0-100).
        face_match_score (float | None): Output of
            modules.face_verification.compare_faces()["similarity_score"]
            (0-100), or None if no selfie was supplied.

    Returns:
        dict: {
            "risk_score": int,              # 0-100
            "risk_band": str,               # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
            "explanation": str,              # human-readable summary
            "contributing_factors": list[dict],  # sorted by weighted contribution, descending
            "excluded_factors": list[str],   # signals that had no data and were left out
            "weights_used": dict,            # the renormalized weights actually applied
        }
    """
    sub_scores: Dict[str, Dict[str, Any]] = {}
    excluded: List[str] = []

    try:
        result = _validation_risk(validation_result)
    except Exception:
        result = None
    if result is not None:
        sub_scores["validation"] = result
    else:
        excluded.append("validation")

    try:
        result = _db_status_risk(db_status)
    except Exception:
        result = None
    if result is not None:
        sub_scores["db_status"] = result
    else:
        excluded.append("db_status")

    try:
        result = _tampering_risk(tampering_score)
    except Exception:
        result = None
    if result is not None:
        sub_scores["tampering"] = result
    else:
        excluded.append("tampering")

    try:
        result = _face_match_risk(face_match_score)
    except Exception:
        result = None
    if result is not None:
        sub_scores["face_match"] = result
    else:
        excluded.append("face_match")

    if not sub_scores:
        return {
            "risk_score": 50,
            "risk_band": "MEDIUM",
            "explanation": "Insufficient data to compute a reliable risk score — treat as medium risk pending manual review.",
            "contributing_factors": [],
            "excluded_factors": excluded,
            "weights_used": {},
        }

    total_weight = sum(RISK_WEIGHTS.get(name, 0.0) for name in sub_scores) or 1.0
    weights_used = {name: RISK_WEIGHTS.get(name, 0.0) / total_weight for name in sub_scores}

    weighted_total = sum(sub_scores[name]["score"] * weights_used[name] for name in sub_scores)
    risk_score = int(round(max(0.0, min(100.0, weighted_total))))

    # Apply hard floors for definitive, binary red flags (see
    # _DB_STATUS_SCORE_FLOOR) — these override the weighted average
    # rather than being just another input to it.
    floor_applied = False
    if db_status and db_status.strip().lower() in _DB_STATUS_SCORE_FLOOR:
        floor = _DB_STATUS_SCORE_FLOOR[db_status.strip().lower()]
        if floor > risk_score:
            risk_score = floor
            floor_applied = True

    if face_match_score is not None and face_match_score < 50.0:
        if risk_score < 65:
            risk_score = 65
            floor_applied = True

    if tampering_score and tampering_score >= 30 and validation_result and validation_result.get("status") == "fail":
        if risk_score < 55:
            risk_score = 55
            floor_applied = True


    risk_band = classify_risk_level(risk_score)

    contributing_factors = sorted(
        (
            {
                "factor": _FACTOR_LABELS.get(name, name),
                "raw_score": round(sub_scores[name]["score"], 1),
                "weight": round(weights_used[name], 3),
                "weighted_score": round(sub_scores[name]["score"] * weights_used[name], 1),
                "detail": sub_scores[name]["detail"],
            }
            for name in sub_scores
        ),
        key=lambda f: f["weighted_score"],
        reverse=True,
    )

    explanation = generate_recommendation(risk_band, contributing_factors)
    if floor_applied:
        explanation += " Score floored due to a registry blacklist hit, which overrides the weighted average."
    if excluded:
        excluded_labels = ", ".join(_FACTOR_LABELS.get(name, name) for name in excluded)
        explanation += f" (No data for: {excluded_labels} — excluded from scoring.)"

    return {
        "risk_score": risk_score,
        "risk_band": risk_band,
        "explanation": explanation,
        "contributing_factors": contributing_factors,
        "excluded_factors": excluded,
        "weights_used": {k: round(v, 3) for k, v in weights_used.items()},
    }
