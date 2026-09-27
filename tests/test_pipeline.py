"""
tests/test_pipeline.py

End-to-end pipeline tests against the synthetic fixtures in
data/test_documents/. Each fixture is run through the same stages
app.py's run_screening_pipeline uses (preprocess -> OCR -> validate ->
DB check -> tamper-detect -> risk score) and the resulting risk band
is asserted against what that kind of document should produce:

    genuine_passport.jpg       -> LOW
        Clean fields, matches a "valid" mock registry entry, no
        tampering signals.

    tampered_passport.jpg      -> HIGH or CRITICAL
        A pasted-in photo with a mismatched compression/noise
        fingerprint, an expiry line rendered in a visibly different
        font (simulating a hand-edited field) showing an expired
        date, a Photoshop EXIF signature, and a passport number
        absent from the registry -- multiple independent red flags,
        deliberately combined the way a real forgery usually presents
        more than one.

    blacklisted_passport.jpg   -> CRITICAL
        Otherwise completely clean, but its passport number matches a
        "blacklisted" mock registry entry. modules.risk_engine floors
        a registry blacklist hit into the CRITICAL band regardless of
        how clean everything else looks -- this fixture exists
        specifically to exercise that floor in isolation.

No selfie is supplied for these fixtures, so face_match_score is
always None; modules.risk_engine excludes it from scoring and
renormalizes the remaining weights, exactly as it would for a real
scan submitted without a selfie.

Run with:
    pytest tests/test_pipeline.py -v
or, without pytest installed:
    python tests/test_pipeline.py
"""

import os
import sys
import unittest
from typing import Any, Dict

# Make the project root importable regardless of the working directory
# this file is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import preprocessing
from modules import ocr, validation, database, tampering, risk_engine, face_verification

TEST_DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "data", "test_documents")


def run_pipeline(image_path: str, document_type: str = "Passport") -> Dict[str, Any]:
    """
    Run a document image through the full non-interactive pipeline
    (everything app.py's run_screening_pipeline does except face
    verification, which needs a second, user-supplied image).

    Args:
        image_path (str): Path to a document image on disk.
        document_type (str): e.g. "Passport".

    Returns:
        dict: {
            "ocr_fields": dict,
            "document_number": str,
            "validation": dict,
            "db_check": dict,
            "tampering": dict,
            "risk": dict,
        }
    """
    image = preprocessing.load_image(image_path)
    steps = preprocessing.run_preprocessing_pipeline(image)
    preprocessed = steps["contrast_enhanced"]
    color_image = steps.get("resized", steps.get("original"))

    ocr_result = ocr.extract_fields(preprocessed, document_type)
    validation_result = validation.validate_document(ocr_result["fields"], document_type)
    document_number = database.get_document_number_from_fields(ocr_result["fields"], document_type)
    db_check_result = database.lookup_document(document_number)
    tampering_result = tampering.detect_tampering(
        color_image, document_type=document_type, original_file_path=image_path
    )
    risk_result = risk_engine.compute_risk_score(
        validation_result,
        db_check_result.get("status"),
        tampering_result.get("tampering_score"),
        None,  # no selfie supplied for these fixtures
    )

    return {
        "ocr_fields": ocr_result["fields"],
        "document_number": document_number,
        "validation": validation_result,
        "db_check": db_check_result,
        "tampering": tampering_result,
        "risk": risk_result,
    }


class TestPipelineRiskBands(unittest.TestCase):
    """Asserts each synthetic fixture lands in its expected risk band."""

    @classmethod
    def setUpClass(cls):
        missing = [
            name
            for name in ("genuine_passport.jpg", "tampered_passport.jpg", "blacklisted_passport.jpg")
            if not os.path.exists(os.path.join(TEST_DOCUMENTS_DIR, name))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing test fixture(s) in {TEST_DOCUMENTS_DIR}: {missing}. "
                f"Run `python data/test_documents/_generate.py` to create them."
            )

    def test_genuine_passport_is_low_risk(self):
        result = run_pipeline(os.path.join(TEST_DOCUMENTS_DIR, "genuine_passport.jpg"))
        self.assertEqual(
            result["db_check"]["status"], "Valid",
            f"Expected the genuine passport's number to match a 'Valid' registry entry, got: {result['db_check']}",
        )
        self.assertEqual(
            result["validation"]["status"], "pass",
            f"Expected clean validation, got: {result['validation']}",
        )
        self.assertEqual(
            result["risk"]["risk_band"], "LOW",
            f"Expected LOW risk for a genuine document, got {result['risk']['risk_band']} "
            f"({result['risk']['risk_score']}). Full risk result: {result['risk']}",
        )

    def test_tampered_passport_is_high_or_critical_risk(self):
        result = run_pipeline(os.path.join(TEST_DOCUMENTS_DIR, "tampered_passport.jpg"))
        self.assertIn(
            result["risk"]["risk_band"], ("HIGH", "CRITICAL"),
            f"Expected HIGH or CRITICAL risk for a tampered document, got "
            f"{result['risk']['risk_band']} ({result['risk']['risk_score']}). "
            f"Tampering flags: {result['tampering']['flags']}. Full risk result: {result['risk']}",
        )
        # The tampering module itself should have flagged something --
        # a HIGH/CRITICAL band driven purely by an unrelated coincidence
        # (e.g. only the registry miss) wouldn't actually demonstrate
        # tampering *detection*.
        self.assertTrue(
            result["tampering"]["flags"],
            f"Expected at least one tampering check to fire, got no flags. Details: {result['tampering']['details']}",
        )

    def test_blacklisted_passport_is_critical_risk(self):
        result = run_pipeline(os.path.join(TEST_DOCUMENTS_DIR, "blacklisted_passport.jpg"))
        self.assertEqual(
            result["db_check"]["status"], "Blacklisted",
            f"Expected the passport number to match a 'Blacklisted' registry entry, got: {result['db_check']}",
        )
        self.assertEqual(
            result["risk"]["risk_band"], "CRITICAL",
            f"Expected CRITICAL risk for a blacklisted document, got {result['risk']['risk_band']} "
            f"({result['risk']['risk_score']}). Full risk result: {result['risk']}",
        )

    def test_mrz_extraction(self):
        sample_mrz_text = (
            "P<FARSI<<AHMADKAL<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
            "Z435R34255ARE03071978M10022020<<<<<<<<<<<<<<<<<<<8\n"
        )
        parsed = ocr.extract_mrz(sample_mrz_text)
        self.assertIsNotNone(parsed, "Expected MRZ parser to return parsed fields")
        self.assertEqual(parsed["Name"], "FARSI AHMADKAL")
        self.assertEqual(parsed["Nationality"], "ARE")
        self.assertEqual(parsed["Gender"], "M")
        self.assertEqual(parsed["Date of Birth"], "03/07/1978")
        self.assertEqual(parsed["Date of Expiry"], "10/02/2020")

    def test_driving_license_validity_extraction(self):
        dl_raw_text = (
            "INDIAN UNION DRIVING LICENCE\n"
            "ISSUED BY ODISHA\n"
            "OD07 20250006115\n"
            "Issue Date Validity ( NT ) Validity ( TR )\n"
            "16-08-2025 01-05-2046\n\n"
            "Name : SUMIT MOHAPATRA\n"
            "Date Of Birth :02-05-2006 Blood Group: Unknown Organ Donor :No\n"
            "Son/Daughter/Wife of : LINGARAJ MOHAPATRA\n"
            "Present Address\n"
            "S/O-LINGARAJ MOHAPATRA,\n"
            "RAGHUPATI NAGAR 1ST LINE Panigrahipentho,\n"
            "Brahmapur Sadar Brahmapur Sadar Ganjam Odisha,760006\n"
        )
        extracted = ocr.parse_generic_id_fields(dl_raw_text)
        fields = extracted[0]
        self.assertEqual(fields["Date of Expiry"], "01-05-2046")
        self.assertEqual(fields["Date of Birth"], "02-05-2006")
        self.assertIn(fields["Document Number"], ("OD07 20250006115", "OD0720250006115"))

        val_result = validation.validate_document(fields, "Driving License")
        self.assertEqual(val_result["status"], "pass")


    def test_driving_license_validity_extraction_with_ocr_gap(self):
        # Simulates Tesseract OCR interleaving neighboring bounding box lines (e.g. "ADPVEH No.(Regn.Numbers) :")
        dl_raw_text = (
            "INDIAN UNION DRIVING LICENCE\n"
            "ISSUED BY ODISHA\n"
            "OD07 20250006115\n"
            "Issue Date Validity ( NT ) Validity ( TR )\n"
            "ADPVEH No.(Regn.Numbers) :\n"
            "16-08-2025 01-05-2046\n\n"
            "Name : SUMIT MOHAPATRA\n"
            "Date Of Birth :02-05-2006 Blood Group: Unknown Organ Donor :No\n"
        )
        extracted = ocr.parse_generic_id_fields(dl_raw_text)
        fields = extracted[0]
        self.assertEqual(fields["Date of Expiry"], "01-05-2046")
        self.assertEqual(fields["Date of Birth"], "02-05-2006")
        self.assertIn(fields["Document Number"], ("OD07 20250006115", "OD0720250006115"))


    def test_face_verification(self):
        import cv2
        import numpy as np

        # Create two synthetic face image crops with skin tone and facial features
        face1 = np.zeros((200, 200, 3), dtype=np.uint8)
        face2 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(face1, (100, 100), 60, (150, 170, 210), -1)
        cv2.circle(face2, (100, 100), 60, (150, 170, 210), -1)
        cv2.circle(face1, (80, 80), 8, (40, 30, 20), -1)
        cv2.circle(face1, (120, 80), 8, (40, 30, 20), -1)
        cv2.circle(face2, (80, 80), 8, (40, 30, 20), -1)
        cv2.circle(face2, (120, 80), 8, (40, 30, 20), -1)

        result = face_verification.compare_faces(face1, face2, db_reference_photo=face1)
        self.assertIsNotNone(result)
        self.assertTrue(result["is_match"])
        self.assertGreaterEqual(result["similarity_score"], 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

