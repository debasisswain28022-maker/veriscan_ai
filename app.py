"""
VeriScan AI — Streamlit entry point.

This is the main application file for the VeriScan AI identity/travel
document screening decision-support system. It wires together the
OCR, validation, tampering-detection, face-verification, and
risk-scoring modules behind a Streamlit UI, and persists results via
the database/blockchain modules.

This file wires the Streamlit UI to the processing pipeline. Upload
handling and preprocessing (utils.preprocessing), OCR field
extraction with per-field confidence scoring (modules.ocr), document
validation (modules.validation), the mock registry "DB Check"
(modules.database), tampering detection (modules.tampering),
selfie-to-document face verification (modules.face_verification),
overall risk scoring (modules.risk_engine), and the hash-chained
audit ledger (modules.blockchain) are all implemented.
"""

from typing import Any, Dict, List, Optional

import streamlit as st

from modules import (
    ocr,
    validation,
    tampering,
    face_verification,
    risk_engine,
    database,
    blockchain,
)
from utils import preprocessing, helpers


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "VeriScan AI"
APP_TAGLINE = "Identity & Travel Document Screening — Decision Support"

NAV_PAGES = ["Upload & Scan", "Dashboard", "Audit Trail", "Settings"]

DOCUMENT_TYPES = ["Passport", "Visa", "National ID", "Driving License", "Permit"]

# Shared badge color mappings (st.badge supports: red, orange, yellow,
# blue, green, violet, gray, primary) — used throughout the result
# panels so severity reads consistently at a glance: green = clean,
# yellow/orange = escalating concern, red = high concern, gray = neutral/unknown.
VALIDATION_BADGE_COLORS = {"pass": "green", "warning": "orange", "fail": "red"}
DB_STATUS_BADGE_COLORS = {"valid": "green", "not found": "gray", "expired": "orange", "blacklisted": "red"}
TAMPERING_BADGE_COLORS = {"low": "green", "medium": "orange", "high": "red"}
RISK_BAND_BADGE_COLORS = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "orange", "CRITICAL": "red"}
FACE_MATCH_BADGE_COLORS = {"match": "green", "no_match": "red", "unknown": "gray"}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def configure_page() -> None:
    """
    Configure Streamlit page settings (title, icon, layout, sidebar state).

    Calls st.set_page_config(...) and applies any global CSS/theming.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🛂",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()


def inject_custom_css() -> None:
    """
    Inject custom CSS for a clean, professional security-tool look:
    a slightly denser layout, a branded sidebar accent, a monospace
    treatment for document/ID numbers, emphasized metrics, a bolder
    primary action button, and colored left-border "cards" for
    severity-graded containers (used by the summary card).

    Loaded once via configure_page(); safe to call multiple times.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        /* Tighten the default top padding so the header sits closer to the top */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Sidebar: subtle branded tint + left accent strip */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 3px solid #1d4ed8;
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #334155;
        }

        /* Monospace treatment for document/ID numbers and codes */
        .veriscan-mono {
            font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        /* Emphasize metric values a bit more than Streamlit's default */
        [data-testid="stMetricValue"] {
            font-weight: 700;
        }

        /* Make the primary action button (Run Screening, Flag for Review) bolder */
        button[kind="primary"] {
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        /* Summary card: a bordered container with a colored left accent,
           set dynamically per risk band via the veriscan-card-{band} class */
        .veriscan-card {
            border-radius: 0.5rem;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-left: 6px solid #94a3b8;
            background-color: rgba(148, 163, 184, 0.06);
        }
        .veriscan-card-low { border-left-color: #16a34a; }
        .veriscan-card-medium { border-left-color: #ca8a04; }
        .veriscan-card-high { border-left-color: #ea580c; }
        .veriscan-card-critical { border-left-color: #dc2626; }

        .veriscan-card-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """
    Initialize all session_state keys used across the app, if not
    already present. Safe to call on every rerun.
    """
    defaults = {
        "current_page": NAV_PAGES[0],
        "document_type": DOCUMENT_TYPES[0],
        "uploaded_document": None,
        "uploaded_face": None,
        "document_temp_path": None,
        "document_validation": None,
        "preprocessing_steps": None,
        "screening_result": None,
        "screening_history": [],
        "flagged_reviews": [],
        "settings": {
            "ocr_engine": "pytesseract",
            "face_backend": "face_recognition",
            "risk_threshold_medium": 0.4,
            "risk_threshold_high": 0.7,
            "enable_blockchain_logging": True,
        },
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Sidebar / navigation
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """
    Render the sidebar: app branding and page navigation.

    Returns:
        str: The currently selected page name (also stored in
            st.session_state["current_page"]).
    """
    with st.sidebar:
        st.markdown(f"## 🛂 {APP_TITLE}")
        st.caption(APP_TAGLINE)
        st.divider()

        selected_page = st.radio(
            "Navigation",
            options=NAV_PAGES,
            index=NAV_PAGES.index(st.session_state["current_page"]),
            label_visibility="collapsed",
        )
        st.session_state["current_page"] = selected_page

        st.divider()
        st.caption("Status")
        st.markdown("- OCR + validation + DB check + tampering + face match + risk score: `implemented`")
        st.markdown("- Audit ledger (hash-chained): `implemented`")
        st.markdown(f"- Records logged: `{len(blockchain.list_ledger_entries())}`")
        st.markdown(f"- Flagged for review: `{len(st.session_state['flagged_reviews'])}`")

    return selected_page


# ---------------------------------------------------------------------------
# Shared header
# ---------------------------------------------------------------------------

def render_header(subtitle: Optional[str] = None) -> None:
    """
    Render the shared page title/header block.

    Args:
        subtitle (str | None): Optional page-specific subtitle shown
            beneath the main title.
    """
    st.title(f"🛂 {APP_TITLE}")
    st.caption(subtitle or APP_TAGLINE)


# ---------------------------------------------------------------------------
# Upload & Scan page
# ---------------------------------------------------------------------------

def render_document_type_selector() -> str:
    """
    Render the document-type selector.

    Returns:
        str: The selected document type.
    """
    document_type = st.selectbox(
        "Document type",
        options=DOCUMENT_TYPES,
        index=DOCUMENT_TYPES.index(st.session_state["document_type"]),
    )
    st.session_state["document_type"] = document_type
    return document_type


def handle_document_upload(document_image: Any) -> Optional[Dict[str, Any]]:
    """
    Handle a newly uploaded document image end-to-end: validate its
    type/size, save it to a temp path, load it, and run it through
    the OCR preprocessing pipeline (grayscale, denoise, deskew,
    contrast enhancement).

    Renders validation errors inline via st.error and stops early if
    the upload fails validation. Also catches load/preprocessing
    failures (e.g. a file that passes the extension/size check but is
    actually corrupted or truncated) the same way, rather than letting
    an unexpected exception crash the page.

    Args:
        document_image: Streamlit UploadedFile for the document image.

    Returns:
        dict | None: {
            "temp_path": str,
            "validation": dict,          # output of preprocessing.validate_upload
            "steps": dict[str, np.ndarray],  # output of preprocessing.run_preprocessing_pipeline
            "preprocessed": np.ndarray,  # == steps["contrast_enhanced"], ready for OCR
        } or None if validation failed or no file was supplied.
    """
    if document_image is None:
        return None

    validation_result = preprocessing.validate_upload(document_image)
    st.session_state["document_validation"] = validation_result

    if not validation_result["is_valid"]:
        for error in validation_result["errors"]:
            st.error(error)
        return None

    temp_path = preprocessing.save_temp_file(document_image)
    st.session_state["document_temp_path"] = temp_path

    try:
        image = preprocessing.load_image(temp_path)
        steps = preprocessing.run_preprocessing_pipeline(image)
    except Exception as exc:
        # A file can pass validate_upload's extension/size checks (it
        # looks like a plausible image) yet still fail to decode or
        # process — a truncated download, a corrupted file, an
        # unsupported color mode, etc. Surface it as a normal error
        # rather than letting it crash the whole page.
        st.error(f"Could not process this image: {exc}. Please try a different file.")
        return None

    st.session_state["preprocessing_steps"] = steps

    return {
        "temp_path": temp_path,
        "validation": validation_result,
        "steps": steps,
        "preprocessed": steps["contrast_enhanced"],
    }


def render_before_after_preview(steps: Dict[str, Any]) -> None:
    """
    Render a side-by-side before/after preview of the preprocessing
    pipeline: the original upload next to the final OCR-ready image,
    with an expander to inspect each intermediate stage.

    Args:
        steps (dict): Output of preprocessing.run_preprocessing_pipeline.
    """
    col_before, col_after = st.columns(2)
    with col_before:
        st.image(
            preprocessing.to_display_image(steps["original"]),
            caption="Before (original upload)",
            use_container_width=True,
        )
    with col_after:
        st.image(
            preprocessing.to_display_image(steps["contrast_enhanced"]),
            caption="After (OCR-ready)",
            use_container_width=True,
        )

    with st.expander("View intermediate preprocessing stages"):
        stage_labels = {
            "resized": "Resized",
            "grayscale": "Grayscale",
            "denoised": "Denoised",
            "deskewed": "Deskewed",
            "contrast_enhanced": "Contrast enhanced (final)",
        }
        stage_cols = st.columns(len(stage_labels))
        for col, (key, label) in zip(stage_cols, stage_labels.items()):
            with col:
                st.image(preprocessing.to_display_image(steps[key]), caption=label, use_container_width=True)


def handle_selfie_upload(selfie_image: Any) -> Optional[Any]:
    """
    Load a selfie/live-capture image — from either the file uploader
    or the webcam widget (st.camera_input, which returns the same
    UploadedFile-like type) — into a standard BGR numpy array for face
    verification.

    Unlike handle_document_upload(), this skips OCR preprocessing and
    temp-file persistence: modules.face_verification only needs raw
    pixel data, and camera_input's UploadedFile doesn't reliably carry
    a filename/extension the way a real upload does, so this loads the
    bytes directly rather than going through preprocessing.validate_upload().

    Args:
        selfie_image: UploadedFile from st.file_uploader or st.camera_input.

    Returns:
        np.ndarray | None: BGR image array, or None if no image was
            supplied or it couldn't be decoded (an inline st.error is
            shown in that case).
    """
    if selfie_image is None:
        return None
    try:
        return preprocessing.load_image(selfie_image)
    except Exception as exc:
        st.error(f"Could not read the selfie/live photo: {exc}")
        return None


def render_upload_section() -> Dict[str, Any]:
    """
    Render the document (and optional selfie/face photo) upload
    widgets, and drive the document through validation, temp-file
    storage, and OCR preprocessing, showing a before/after preview.
    The selfie can come from a file upload or the webcam.

    Returns:
        dict: {
            "document_image": UploadedFile | None,
            "face_image": UploadedFile | None,
            "document_processed": dict | None,  # output of handle_document_upload
            "face_processed": np.ndarray | None,  # output of handle_selfie_upload
        }
    """
    col1, col2 = st.columns(2)

    with col1:
        document_image = st.file_uploader(
            "Upload document image",
            type=sorted(preprocessing.ALLOWED_EXTENSIONS),
            key="document_uploader",
            help=(
                f"Upload a scan or photo of the passport, visa, ID, license, or permit. "
                f"Max size: {preprocessing.MAX_FILE_SIZE_MB:.0f} MB."
            ),
        )

    with col2:
        selfie_source = st.radio(
            "Selfie source", ["Upload photo", "Use webcam"], horizontal=True, key="selfie_source"
        )
        if selfie_source == "Use webcam":
            face_image = st.camera_input(
                "Take a live photo", key="face_camera_input", help="Optional — used for document-to-face match verification."
            )
        else:
            face_image = st.file_uploader(
                "Upload selfie / live photo (optional)",
                type=sorted(preprocessing.ALLOWED_EXTENSIONS),
                key="face_uploader",
                help="Optional — used for document-to-face match verification.",
            )
        if face_image is not None and selfie_source != "Use webcam":
            # camera_input already renders its own live preview; avoid a duplicate.
            st.image(face_image, caption="Selfie preview", use_container_width=True)

    st.session_state["uploaded_document"] = document_image
    st.session_state["uploaded_face"] = face_image

    document_processed = None
    with col1:
        if document_image is not None:
            with st.spinner("Validating and preprocessing document..."):
                document_processed = handle_document_upload(document_image)

    face_processed = handle_selfie_upload(face_image)

    if document_processed is not None:
        st.divider()
        st.markdown("**Preprocessing preview**")
        render_before_after_preview(document_processed["steps"])

    return {
        "document_image": document_image,
        "face_image": face_image,
        "document_processed": document_processed,
        "face_processed": face_processed,
    }


def run_screening_pipeline(uploads: Dict[str, Any], options: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Orchestrate the end-to-end screening pipeline for a submitted
    document (and optional face photo): preprocess -> OCR field
    extraction -> validation -> DB check -> tampering detection ->
    face verification -> risk scoring, with each stage's output
    feeding the next.

    Fault isolation: every stage below already has its own internal
    "never crash" contract (see each module's docstring), but this
    orchestrator adds a second layer of defense — each stage call is
    wrapped individually, so an unexpected failure in any one stage
    (a bug, an unsupported input shape, a missing dependency) cannot
    take down the stages after it. A failed stage is recorded in the
    returned "pipeline_warnings" list and substituted with a neutral
    fallback value (e.g. an empty OCR result, a "warning"-status
    validation result) so downstream stages that depend on it still
    have something well-shaped to work with, and the UI can show
    whatever partial results *did* succeed rather than nothing at all.
    Stages that don't depend on an earlier stage's output (tampering
    and face verification both work directly from the document image,
    not from OCR's fields) still run even if an earlier stage failed.

    Expected stages, each behind its own st.spinner:
        1. Preprocess images (utils.preprocessing) — see
           handle_document_upload(); the preprocessed image is
           available at uploads["document_processed"]["preprocessed"].
        2. Extract text via OCR (modules.ocr.extract_fields).
        3. Validate extracted fields (modules.validation.validate_document),
           fed the OCR fields from step 2.
        4. Check the document number against the mock registry
           (modules.database.lookup_document — "DB Check"), fed the
           document number pulled from the OCR fields in step 2.
        5. Run tampering/forgery detection (modules.tampering.detect_tampering)
           directly on the document image.
        6. Run face verification, if a selfie was supplied
           (modules.face_verification.compare_faces), reusing the same
           document image as step 5.
        7. Aggregate signals into a risk score (modules.risk_engine.compute_risk_score),
           fed the validation status, DB status, tampering score, and
           face-match score from steps 3, 4, 5, and 6.
        8. Persist the result (modules.database, modules.blockchain) —
           not yet implemented; its result key is set to None.

    Args:
        uploads (dict): Output of render_upload_section(), including
            the preprocessed document image under
            uploads["document_processed"]["preprocessed"] and the
            loaded selfie array under uploads["face_processed"].
        options (dict): Selected options (e.g. document type, settings).

    Returns:
        dict | None: None if no document was successfully preprocessed
            (the one failure mode that genuinely blocks the whole
            pipeline, since every later stage needs the document
            image). Otherwise:
            {
                "document_type": str,
                "ocr_fields": dict,               # e.g. {"Name": ..., "Passport Number": ..., ...}
                "ocr_field_confidence": dict,      # field name -> int 0-100
                "ocr_low_confidence_fields": list[str],  # field names flagged for manual review
                "ocr_raw_text": str,
                "ocr_confidence": float,
                "ocr_bounding_boxes": list[dict],
                "ocr_error": str | None,
                "validation": dict,          # {"document_type", "status", "issues"} from modules.validation
                "db_check": dict,            # {"status", "record", "error"} from modules.database.lookup_document
                "db_check_document_number": str,  # the number that was looked up
                "tampering": dict,           # {"tampering_score", "risk_level", "flags", "details"} from modules.tampering
                "face_match": dict | None,   # output of modules.face_verification.compare_faces, or None if no selfie was supplied
                "risk": dict,                # {"risk_score", "risk_band", "explanation", "contributing_factors", ...} from modules.risk_engine
                "pipeline_warnings": list[dict],  # [{"stage", "error"}, ...] for any stage that failed unexpectedly
            }
    """
    document_processed = uploads.get("document_processed")
    if document_processed is None:
        return None

    preprocessed_image = document_processed["preprocessed"]
    document_type = options.get("document_type", DOCUMENT_TYPES[0])
    color_image = document_processed["steps"].get("resized", document_processed["steps"].get("original"))

    pipeline_warnings: List[Dict[str, str]] = []

    def _run_stage(stage_name: str, spinner_text: str, fn, fallback_factory) -> Any:
        """
        Run one pipeline stage under a spinner, catching any exception
        so a single stage's failure can't take down the stages after
        it. On failure, records {"stage", "error"} in
        pipeline_warnings and returns fallback_factory(exc) instead of
        propagating.
        """
        with st.spinner(spinner_text):
            try:
                return fn()
            except Exception as exc:
                pipeline_warnings.append({"stage": stage_name, "error": str(exc)})
                return fallback_factory(exc)

    ocr_result = _run_stage(
        "OCR extraction", "Running OCR extraction...",
        lambda: ocr.extract_fields(preprocessed_image, document_type),
        lambda exc: {
            "fields": {}, "field_confidence": {}, "low_confidence_fields": [],
            "raw_text": "", "ocr_confidence": 0.0, "bounding_boxes": [],
            "ocr_error": f"OCR stage failed unexpectedly: {exc}",
        },
    )

    validation_result = _run_stage(
        "Validation", "Validating extracted fields...",
        lambda: validation.validate_document(ocr_result["fields"], document_type),
        lambda exc: {
            "document_type": document_type, "status": "warning",
            "issues": [{"field": "_pipeline", "severity": "warning",
                        "message": f"Validation stage failed unexpectedly: {exc}"}],
        },
    )

    def _run_db_check():
        number = database.get_document_number_from_fields(ocr_result["fields"], document_type)
        return number, database.lookup_document(number)

    document_number, db_check_result = _run_stage(
        "Registry check", "Checking document registry...",
        _run_db_check,
        lambda exc: (None, {"status": "Not Found", "record": None, "error": f"DB check stage failed unexpectedly: {exc}"}),
    )

    tampering_result = _run_stage(
        "Tampering detection", "Analyzing document for signs of tampering...",
        lambda: tampering.detect_tampering(
            color_image, document_type=document_type, original_file_path=document_processed.get("temp_path")
        ),
        lambda exc: {"tampering_score": 0, "risk_level": "low", "flags": [], "details": {}},
    )

    face_match_result = None
    selfie_array = uploads.get("face_processed")
    if selfie_array is not None:
        face_match_result = _run_stage(
            "Face verification", "Verifying face match...",
            lambda: face_verification.compare_faces(color_image, selfie_array),
            lambda exc: {
                "is_match": None, "similarity_score": None, "threshold": 50.0,
                "backend_used": None, "raw_distance": None,
                "document_face_count": 0, "selfie_face_count": 0,
                "multiple_faces_flagged": False, "no_face_detected": False,
                "error": f"Face verification stage failed unexpectedly: {exc}", "notes": [],
            },
        )

    risk_result = _run_stage(
        "Risk scoring", "Computing overall risk score...",
        lambda: risk_engine.compute_risk_score(
            validation_result,
            db_check_result.get("status"),
            tampering_result.get("tampering_score"),
            face_match_result.get("similarity_score") if face_match_result else None,
        ),
        lambda exc: {
            "risk_score": 50, "risk_band": "MEDIUM",
            "explanation": "Risk scoring could not be completed due to an unexpected error; treat as medium risk pending manual review.",
            "contributing_factors": [], "excluded_factors": [], "weights_used": {},
        },
    )

    return {
        "document_type": document_type,
        "ocr_fields": ocr_result["fields"],
        "ocr_field_confidence": ocr_result.get("field_confidence", {}),
        "ocr_low_confidence_fields": ocr_result.get("low_confidence_fields", []),
        "ocr_raw_text": ocr_result["raw_text"],
        "ocr_confidence": ocr_result["ocr_confidence"],
        "ocr_bounding_boxes": ocr_result["bounding_boxes"],
        "ocr_error": ocr_result["ocr_error"],
        "validation": validation_result,
        "db_check": db_check_result,
        "db_check_document_number": document_number,
        "tampering": tampering_result,
        "face_match": face_match_result,
        "risk": risk_result,
        "pipeline_warnings": pipeline_warnings,
    }


def render_field_confidence_table(
    fields: Dict[str, Any], field_confidence: Dict[str, int], low_confidence_fields: Optional[List[str]] = None
) -> None:
    """
    Render each extracted field next to its 0-100 confidence score,
    visually highlighting fields flagged for manual review.

    Args:
        fields (dict): Field name -> extracted value (e.g. from
            ocr.extract_fields()["fields"]). May instead be a single
            {"note": ...} dict for document types without a parser yet.
        field_confidence (dict): Field name -> int confidence (0-100),
            from ocr.extract_fields()["field_confidence"].
        low_confidence_fields (list[str] | None): Field names below
            ocr.LOW_CONFIDENCE_THRESHOLD, from
            ocr.extract_fields()["low_confidence_fields"].
    """
    if not fields:
        st.info("No fields extracted.")
        return

    if "note" in fields and not field_confidence:
        st.info(fields["note"])
        return

    low_confidence_fields = set(low_confidence_fields or [])

    header_name, header_value, header_conf = st.columns([2, 3, 3])
    header_name.markdown("**Field**")
    header_value.markdown("**Extracted value**")
    header_conf.markdown("**Confidence**")

    for field_name, value in fields.items():
        confidence = field_confidence.get(field_name, 0)
        flagged = field_name in low_confidence_fields or confidence < ocr.LOW_CONFIDENCE_THRESHOLD

        col_name, col_value, col_conf = st.columns([2, 3, 3])
        with col_name:
            st.markdown(f"**{field_name}**")
        with col_value:
            if flagged:
                st.markdown(f":red[{value}]")
            else:
                st.markdown(str(value))
        with col_conf:
            bar_label = f"{confidence}%" + ("  ⚠️ Needs review" if flagged else "")
            st.progress(confidence / 100.0, text=bar_label)

    if low_confidence_fields:
        st.warning(
            f"⚠️ Flagged for manual review (confidence below {ocr.LOW_CONFIDENCE_THRESHOLD}%): "
            f"{', '.join(sorted(low_confidence_fields))}"
        )


def render_validation_panel(validation_result: Dict[str, Any]) -> None:
    """
    Render the document-validation status and issue list: an overall
    pass/warning/fail badge, followed by each issue color-coded by
    severity (error vs. warning).

    Args:
        validation_result (dict): Output of
            modules.validation.validate_document(), i.e.
            {"document_type", "status", "issues"}.
    """
    status = validation_result.get("status", "warning")
    issues = validation_result.get("issues", [])

    status_labels = {"pass": "PASS", "warning": "WARNING", "fail": "FAIL"}
    status_icons = {"pass": "✅", "warning": "⚠️", "fail": "❌"}
    st.badge(
        status_labels.get(status, status.upper()),
        icon=status_icons.get(status, "⚠️"),
        color=VALIDATION_BADGE_COLORS.get(status, "orange"),
    )

    if not issues:
        st.caption("No issues found.")
        return

    errors = [issue for issue in issues if issue.get("severity") == "error"]
    warnings = [issue for issue in issues if issue.get("severity") == "warning"]

    for issue in errors:
        st.error(f"**{issue['field']}** — {issue['message']}")
    for issue in warnings:
        st.warning(f"**{issue['field']}** — {issue['message']}")


def render_db_check_panel(db_check: Dict[str, Any], document_number: str) -> None:
    """
    Render the mock registry "DB Check" result: whether the extracted
    document number matches a Valid, Expired, or Blacklisted record in
    modules.database's synthetic registry, or wasn't found at all.

    Args:
        db_check (dict): Output of modules.database.lookup_document(),
            i.e. {"status", "record", "error"}.
        document_number (str): The document number that was looked up,
            for display context.
    """
    status = db_check.get("status", "Not Found")
    record = db_check.get("record")

    status_icons = {"Valid": "✅", "Expired": "⏳", "Blacklisted": "🚫", "Not Found": "❓"}
    badge_color = DB_STATUS_BADGE_COLORS.get(status.lower(), "gray")

    if document_number and document_number != ocr.NOT_DETECTED:
        st.markdown(f"Registry status for <span class='veriscan-mono'>{document_number}</span>:", unsafe_allow_html=True)
    else:
        st.caption("No document number was extracted to look up.")
    st.badge(status.upper(), icon=status_icons.get(status, "❓"), color=badge_color)

    if db_check.get("error"):
        st.caption(f"⚠️ {db_check['error']}")

    if record:
        col1, col2, col3 = st.columns(3)
        col1.metric("Name on file", record.get("full_name") or "—")
        col2.metric("Nationality on file", record.get("nationality") or "—")
        col3.metric("Registry expiry", record.get("expiry_date") or "—")
        if record.get("notes"):
            st.caption(f"Notes: {record['notes']}")

    st.caption("This registry is a synthetic mock dataset for demonstration only — not a real government or issuer database.")


def render_tampering_panel(tampering_result: Dict[str, Any]) -> None:
    """
    Render the tampering-detection result: an overall 0-100 score with
    risk-level badge, followed by a breakdown of each individual check
    (photo substitution, text manipulation, stamp forgery, metadata)
    showing whether it fired and why.

    Args:
        tampering_result (dict): Output of modules.tampering.detect_tampering(),
            i.e. {"tampering_score", "risk_level", "flags", "details"}.
    """
    score = tampering_result.get("tampering_score", 0)
    risk_level = tampering_result.get("risk_level", "low")
    flags = tampering_result.get("flags", [])
    details = tampering_result.get("details", {})

    risk_icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}

    col_score, col_badge = st.columns([1, 2])
    with col_score:
        st.metric("Tampering score", f"{score}/100")
    with col_badge:
        st.badge(
            f"{risk_level.upper()} RISK",
            icon=risk_icons.get(risk_level, "🟡"),
            color=TAMPERING_BADGE_COLORS.get(risk_level, "orange"),
        )

    if flags:
        st.caption(f"Checks that fired: {', '.join(flag.replace('_', ' ').title() for flag in flags)}")

    check_labels = {
        "photo_substitution": "🖼️ Photo Substitution",
        "text_manipulation": "🔤 Text Manipulation",
        "stamp_forgery": "🔏 Stamp Forgery",
        "metadata": "🗂️ Metadata Analysis",
    }

    for check_key, check_label in check_labels.items():
        detail = details.get(check_key)
        if detail is None:
            continue
        is_suspicious = detail.get("is_suspicious", False)
        confidence = detail.get("confidence", 0.0)
        icon = "⚠️" if is_suspicious else "✅"
        with st.expander(f"{icon} {check_label} — {'Flagged' if is_suspicious else 'Clear'} ({confidence:.0%} confidence)"):
            for note in detail.get("notes", []):
                st.caption(note)

    st.caption(
        "Tampering checks are heuristic signals for manual review, not proof of forgery — "
        "low-quality scans or photos can also trigger them."
    )


def render_face_match_panel(face_match: Dict[str, Any]) -> None:
    """
    Render the face-verification result: similarity score, match/
    no-match badge, and edge-case handling (no selfie, no face
    detected in either image, multiple faces detected).

    Args:
        face_match (dict): Output of modules.face_verification.compare_faces().
    """
    if face_match.get("error"):
        st.error(f"Face verification could not be completed: {face_match['error']}")
        return

    if face_match.get("no_face_detected"):
        for note in face_match.get("notes", []):
            st.warning(note)
        st.caption(
            f"Document photo faces detected: {face_match.get('document_face_count', 0)} | "
            f"Selfie faces detected: {face_match.get('selfie_face_count', 0)}"
        )
        return

    similarity = face_match.get("similarity_score")
    threshold = face_match.get("threshold", 50.0)
    is_match = face_match.get("is_match")

    col_score, col_badge = st.columns([1, 2])
    with col_score:
        st.metric("Similarity score", f"{similarity:.1f}%" if similarity is not None else "—")
    with col_badge:
        if is_match:
            st.badge(f"MATCH (threshold: {threshold:.0f}%)", icon="✅", color=FACE_MATCH_BADGE_COLORS["match"])
        else:
            st.badge(f"NO MATCH (threshold: {threshold:.0f}%)", icon="❌", color=FACE_MATCH_BADGE_COLORS["no_match"])

    st.caption(
        f"Backend: {face_match.get('backend_used', '—')} | "
        f"Document faces: {face_match.get('document_face_count', 0)} | "
        f"Selfie faces: {face_match.get('selfie_face_count', 0)}"
    )

    for note in face_match.get("notes", []):
        st.warning(note)


def render_risk_panel(risk_result: Dict[str, Any]) -> None:
    """
    Render the final risk-decision panel: a large color-coded band
    badge, the 0-100 score, the human-readable explanation, and a
    breakdown of contributing factors ranked by weighted contribution.

    This is the last stop in the pipeline — it's presented as
    decision *support*, not an automated verdict.

    Args:
        risk_result (dict): Output of modules.risk_engine.compute_risk_score(),
            i.e. {"risk_score", "risk_band", "explanation",
            "contributing_factors", "excluded_factors", "weights_used"}.
    """
    score = risk_result.get("risk_score", 0)
    band = risk_result.get("risk_band", "MEDIUM")
    explanation = risk_result.get("explanation", "")
    factors = risk_result.get("contributing_factors", [])
    excluded = risk_result.get("excluded_factors", [])

    band_icons = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}

    col_score, col_band = st.columns([1, 2])
    with col_score:
        st.metric("Risk score", f"{score}/100")
    with col_band:
        st.badge(
            f"{band} RISK",
            icon=band_icons.get(band, "🟡"),
            color=RISK_BAND_BADGE_COLORS.get(band, "orange"),
        )

    st.markdown(f"**Recommendation:** {explanation}")

    if factors:
        st.caption("Contributing factors (ranked by impact on the score):")
        for factor in factors:
            st.markdown(
                f"- **{factor['factor']}** — {factor['detail']} "
                f"(contributed {factor['weighted_score']:.1f} pts, {factor['weight']:.0%} weight)"
            )

    if excluded:
        st.caption(f"Not included in scoring (no data): {', '.join(excluded)}.")

    st.caption(
        "This is a decision-support score for human review, not an automated accept/reject decision."
    )


def render_summary_card(result: Dict[str, Any]) -> None:
    """
    Render a top-of-page summary card: the overall risk band and score
    as the headline (color-coded via the veriscan-card-{band} CSS
    class), plus a compact one-line status across all five modules.
    This is the first thing a reviewer should see — the expandable
    per-module sections below are for drilling into *why*.

    Args:
        result (dict): Output of run_screening_pipeline().
    """
    risk = result.get("risk") or {}
    band = risk.get("risk_band", "MEDIUM")
    score = risk.get("risk_score", "—")
    band_class = f"veriscan-card-{band.lower()}"
    band_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(band, "🟡")

    validation_status = (result.get("validation") or {}).get("status", "—")
    db_status = (result.get("db_check") or {}).get("status", "—")
    tampering_level = (result.get("tampering") or {}).get("risk_level", "—")

    face_match = result.get("face_match")
    if face_match is None:
        face_text = "not checked"
    elif face_match.get("is_match") is True:
        face_text = "match"
    elif face_match.get("is_match") is False:
        face_text = "no match"
    else:
        face_text = "inconclusive"

    document_type = result.get("document_type", "—")
    document_number = result.get("db_check_document_number") or None
    subtitle = document_type + (f" · <span class='veriscan-mono'>{document_number}</span>" if document_number else "")

    st.markdown(
        f"""
        <div class="veriscan-card {band_class}">
            <div class="veriscan-card-title">Screening Summary — {subtitle}</div>
            <div style="font-size:1.6rem; font-weight:700; margin-bottom:0.35rem;">
                {band_emoji} {band} RISK
                <span style="font-weight:400; font-size:1rem; color:#64748b;">&nbsp;({score}/100)</span>
            </div>
            <div style="font-size:0.92rem; color:#475569;">
                Validation: <b>{str(validation_status).upper()}</b> &nbsp;·&nbsp;
                Registry: <b>{db_status}</b> &nbsp;·&nbsp;
                Tampering: <b>{str(tampering_level).upper()}</b> &nbsp;·&nbsp;
                Face match: <b>{face_text}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flag_for_review_button(result: Dict[str, Any]) -> None:
    """
    Show a "Flag for Manual Review" action when the overall risk band
    is HIGH or CRITICAL, letting a reviewer explicitly mark this scan
    for follow-up. Does nothing for LOW/MEDIUM scans. Flagged scans
    are recorded in st.session_state["flagged_reviews"] (reflected in
    the sidebar's "Flagged for review" count) and the flag is stamped
    onto the result itself so re-rendering after a rerun doesn't
    duplicate it.

    Args:
        result (dict): Output of run_screening_pipeline().
    """
    risk = result.get("risk") or {}
    band = risk.get("risk_band")
    if band not in ("HIGH", "CRITICAL"):
        return

    if result.get("_flagged"):
        st.success(f"🚩 Flagged for manual review at {result.get('_flagged_at', 'an earlier time')}.")
        return

    st.warning(f"This scan scored **{band}** risk — consider flagging it for manual review.")
    if st.button("🚩 Flag for Manual Review", type="primary", key="flag_for_review_button"):
        flagged_at = helpers.get_timestamp()
        st.session_state["flagged_reviews"].append({
            "document_type": result.get("document_type"),
            "document_number": result.get("db_check_document_number"),
            "risk_band": band,
            "risk_score": risk.get("risk_score"),
            "flagged_at": flagged_at,
        })
        result["_flagged"] = True
        result["_flagged_at"] = flagged_at
        st.rerun()


def render_result_panels(result: Optional[Dict[str, Any]]) -> None:
    """
    Render each pipeline module's result as its own expandable section
    (OCR, Validation, DB Check, Tampering, Face Match, Risk). Each
    expander's label includes a quick-glance status, and sections that
    need attention (a failed check, a low match, a non-LOW risk band)
    auto-expand while clean ones stay collapsed — so a reviewer's eye
    goes straight to what matters. Shows placeholder/empty states
    until `result` is populated by run_screening_pipeline.

    Args:
        result (dict | None): Output of run_screening_pipeline(), or
            None if no screening has been run yet.
    """
    ocr_label = "📝 OCR Fields"
    if result and result.get("ocr_fields"):
        ocr_label += f" — {result.get('ocr_confidence', 0.0):.0%} confidence"
    with st.expander(ocr_label, expanded=bool(result and result.get("ocr_low_confidence_fields"))):
        if result and result.get("ocr_fields"):
            if result.get("ocr_error"):
                st.warning(f"OCR completed with a fallback: {result['ocr_error']}")
            render_field_confidence_table(
                result["ocr_fields"],
                result.get("ocr_field_confidence", {}),
                result.get("ocr_low_confidence_fields", []),
            )
            with st.expander("View raw OCR text"):
                st.text(result.get("ocr_raw_text", ""))
        else:
            st.info("No OCR results yet. Run a scan to extract document fields.")

    validation_status = result.get("validation", {}).get("status") if result else None
    validation_label = "✅ Validation" + (f" — {validation_status.upper()}" if validation_status else "")
    with st.expander(validation_label, expanded=bool(validation_status and validation_status != "pass")):
        if result and result.get("validation"):
            render_validation_panel(result["validation"])
        else:
            st.info("No validation results yet. Run a scan to check field validity.")

    db_status = result.get("db_check", {}).get("status") if result else None
    db_label = "🗄️ DB Check" + (f" — {db_status}" if db_status else "")
    with st.expander(db_label, expanded=bool(db_status and db_status != "Valid")):
        if result and result.get("db_check"):
            render_db_check_panel(result["db_check"], result.get("db_check_document_number", ""))
        else:
            st.info("No registry check yet. Run a scan to look up the document number.")

    tampering_level = result.get("tampering", {}).get("risk_level") if result else None
    tampering_score = result.get("tampering", {}).get("tampering_score") if result else None
    tampering_label = "🔍 Tampering" + (f" — {tampering_level.upper()} ({tampering_score}/100)" if tampering_level else "")
    with st.expander(tampering_label, expanded=bool(tampering_level and tampering_level != "low")):
        if result and result.get("tampering") is not None:
            render_tampering_panel(result["tampering"])
        else:
            st.info("No tampering analysis yet. Run a scan to check for signs of manipulation.")

    face_match = result.get("face_match") if result else None
    face_label = "🧑‍🤝‍🧑 Face Match"
    face_needs_attention = False
    if face_match:
        if face_match.get("is_match") is True:
            face_label += " — MATCH"
        elif face_match.get("is_match") is False:
            face_label += " — NO MATCH"
            face_needs_attention = True
        else:
            face_label += " — N/A"
            face_needs_attention = True
    with st.expander(face_label, expanded=face_needs_attention):
        if face_match is not None:
            render_face_match_panel(face_match)
        else:
            st.info("No face match yet. Upload or capture a selfie and run a scan to compare against the document photo.")

    risk_band = result.get("risk", {}).get("risk_band") if result else None
    risk_score = result.get("risk", {}).get("risk_score") if result else None
    risk_label = "⚠️ Risk Score" + (f" — {risk_band} ({risk_score}/100)" if risk_band else "")
    with st.expander(risk_label, expanded=bool(risk_band and risk_band != "LOW")):
        if result and result.get("risk") is not None:
            render_risk_panel(result["risk"])
        else:
            st.info("No risk assessment yet. Run a scan to generate an overall risk verdict.")


def render_upload_scan_page() -> None:
    """
    Render the full "Upload & Scan" page: document type selector,
    upload widgets, a scan trigger, and the placeholder result panels.
    """
    render_header("Upload a document to begin a screening session.")

    render_document_type_selector()
    uploads = render_upload_section()

    st.divider()

    scan_clicked = st.button(
        "▶️ Run Screening",
        type="primary",
        disabled=uploads.get("document_processed") is None,
        use_container_width=False,
        help="Upload a valid document image to enable screening." if uploads.get("document_processed") is None else None,
    )

    if scan_clicked:
        options = {
            "document_type": st.session_state["document_type"],
            **st.session_state["settings"],
        }
        result = run_screening_pipeline(uploads, options)
        st.session_state["screening_result"] = result
        if result is None:
            st.warning("Could not run screening — please re-upload a valid document.")
        else:
            if st.session_state["settings"].get("enable_blockchain_logging", True):
                ledger_entry = blockchain.add_record(
                    document_id=result.get("db_check_document_number") or "Not Detected",
                    risk_score=result["risk"]["risk_score"],
                    decision=result["risk"]["risk_band"],
                )
                result["ledger_entry"] = ledger_entry

            pipeline_warnings = result.get("pipeline_warnings", [])
            if pipeline_warnings:
                failed_stages = ", ".join(w["stage"] for w in pipeline_warnings)
                st.warning(
                    f"Screening completed with partial results — the following stage(s) hit an "
                    f"unexpected error and fell back to a neutral default: {failed_stages}. "
                    f"Other stages still ran normally; treat this scan as needing manual review."
                )
                with st.expander("Stage error details"):
                    for w in pipeline_warnings:
                        st.code(f"[{w['stage']}] {w['error']}")
            elif result.get("ocr_error"):
                st.info(f"OCR note: {result['ocr_error']}")
            else:
                st.success("Screening complete: OCR, validation, registry check, tampering analysis, face match, and risk scoring all ran.")

    st.divider()
    st.subheader("Results")

    screening_result = st.session_state["screening_result"]
    if screening_result:
        render_summary_card(screening_result)
        render_flag_for_review_button(screening_result)

    render_result_panels(screening_result)


# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------

def render_dashboard_page() -> None:
    """
    Render the "Dashboard" page: summary metrics and history of past
    screening records, sourced from the hash-chained audit ledger
    (modules.blockchain) so figures reflect every scan actually logged,
    not just this browser session's in-memory state.
    """
    render_header("Overview of screening activity.")

    entries = blockchain.list_ledger_entries()
    total = len(entries)
    high_risk = sum(1 for e in entries if str(e.get("decision", "")).upper() in ("HIGH", "CRITICAL"))
    avg_score = round(sum(e.get("risk_score", 0) or 0 for e in entries) / total, 1) if total else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total screened", total or "—")
    col2.metric("High/Critical risk", high_risk if total else "—")
    col3.metric("Flagged for review", len(st.session_state["flagged_reviews"]) or "—")
    col4.metric("Avg. risk score", avg_score if avg_score is not None else "—")

    st.divider()
    render_history()


def render_history() -> None:
    """
    Render a table of past screening records pulled from the
    hash-chained audit ledger (modules.blockchain), newest first.
    """
    entries = blockchain.list_ledger_entries()
    if not entries:
        st.info("No screening records yet. Completed scans will appear here.")
        return

    table_rows = [
        {
            "Timestamp": entry.get("timestamp", "—"),
            "Document ID": entry.get("document_id", "—"),
            "Risk Score": entry.get("risk_score", "—"),
            "Decision": entry.get("decision", "—"),
        }
        for entry in reversed(entries)
    ]
    st.dataframe(table_rows, use_container_width=True)


# ---------------------------------------------------------------------------
# Audit Trail page
# ---------------------------------------------------------------------------

def render_audit_trail_page() -> None:
    """
    Render the "Audit Trail" page: every past scan recorded in the
    hash-chained ledger (modules.blockchain), newest first, alongside
    a chain-integrity indicator (re-hashes every entry to confirm
    nothing has been edited, inserted, or removed since it was logged).
    """
    render_header("Tamper-evident log of screening decisions.")

    integrity = blockchain.verify_chain()

    if integrity["total_records"] == 0:
        st.info("No ledger entries yet. Run a screening on the Upload & Scan page to add the first record.")
        return

    if integrity["is_valid"]:
        st.success(f"✅ Chain integrity verified — {integrity['total_records']} record(s), no tampering detected.")
    else:
        st.error(
            f"🚨 CHAIN INTEGRITY FAILURE at entry {integrity['broken_at_index']}: {integrity['reason']} "
            f"Records after this point cannot be trusted without investigation."
        )

    with st.expander("How this is verified"):
        st.caption(
            "Each record stores a hash computed over its own fields plus the previous record's "
            "hash, forming a chain. Editing, deleting, or reordering any past record changes its "
            "hash and breaks the link to the record after it — which is exactly what this check "
            "re-derives from scratch on every page load, rather than trusting a stored 'valid' flag."
        )

    st.divider()

    entries = list(reversed(blockchain.list_ledger_entries()))
    band_icons = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}

    for entry in entries:
        decision = str(entry.get("decision", "—"))
        icon = band_icons.get(decision.upper(), "⚪")
        label = f"{icon} #{entry.get('index')} — {entry.get('document_id', '—')} — {decision} ({entry.get('risk_score', '—')}/100)"
        with st.expander(label):
            st.markdown(f"**Timestamp:** {entry.get('timestamp', '—')}")
            st.markdown(f"**Document ID:** `{entry.get('document_id', '—')}`")
            st.markdown(f"**Risk score:** {entry.get('risk_score', '—')}/100 — **{decision}**")
            st.caption(f"Hash: `{entry.get('hash', '—')}`")
            st.caption(f"Prev hash: `{entry.get('prev_hash', '—')}`")


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------

def render_settings_page() -> None:
    """
    Render the "Settings" page: configurable pipeline options
    (OCR engine, face-verification backend, risk thresholds, and
    audit-logging toggle). Updates st.session_state["settings"].
    """
    render_header("Configure screening pipeline behavior.")

    settings = st.session_state["settings"]

    settings["ocr_engine"] = st.selectbox(
        "OCR engine", options=["pytesseract", "easyocr"],
        index=["pytesseract", "easyocr"].index(settings["ocr_engine"]),
    )
    settings["face_backend"] = st.selectbox(
        "Face verification backend", options=["face_recognition", "deepface"],
        index=["face_recognition", "deepface"].index(settings["face_backend"]),
    )

    st.divider()
    st.subheader("Risk thresholds")
    settings["risk_threshold_medium"] = st.slider(
        "Medium risk threshold", 0.0, 1.0, settings["risk_threshold_medium"]
    )
    settings["risk_threshold_high"] = st.slider(
        "High risk threshold", 0.0, 1.0, settings["risk_threshold_high"]
    )

    st.divider()
    settings["enable_blockchain_logging"] = st.toggle(
        "Enable audit ledger logging", value=settings["enable_blockchain_logging"]
    )

    st.session_state["settings"] = settings
    st.caption("Settings are stored for this session only until persistence is implemented.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_results(result: Optional[Dict[str, Any]]) -> None:
    """
    Render the screening result in the main panel (extracted fields,
    validation flags, tampering heatmap/notes, face match score,
    overall risk verdict, and a downloadable report link).

    Currently delegates to render_result_panels() for the placeholder
    tabbed layout; report export/heatmap rendering is added once
    run_screening_pipeline is implemented.

    Args:
        result (dict | None): Output of run_screening_pipeline().
    """
    render_result_panels(result)


def main() -> None:
    """
    Application entry point. Configures the page, initializes session
    state, renders the sidebar navigation, and dispatches to the
    selected page's render function.
    """
    configure_page()
    init_session_state()

    page = render_sidebar()

    page_renderers = {
        "Upload & Scan": render_upload_scan_page,
        "Dashboard": render_dashboard_page,
        "Audit Trail": render_audit_trail_page,
        "Settings": render_settings_page,
    }
    page_renderers[page]()


if __name__ == "__main__":
    main()
