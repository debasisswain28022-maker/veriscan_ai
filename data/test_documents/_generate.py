"""
data/test_documents/_generate.py

One-off generator for the three synthetic passport fixtures in this
directory, kept here for reproducibility (so the fixtures can be
regenerated or tweaked without hand-editing binary JPEGs). Not part of
the test suite itself — run manually if the fixtures ever need
regenerating:

    python data/test_documents/_generate.py

Produces:
    genuine_passport.jpg       — clean, valid, matches a "valid" mock
                                  registry entry -> expect LOW risk.
    tampered_passport.jpg      — pasted-in photo with a mismatched
                                  compression/noise fingerprint,
                                  inconsistent text baseline/size on
                                  the expiry line, an expired date, a
                                  Photoshop EXIF signature, and a
                                  passport number that isn't in the
                                  registry -> expect HIGH/CRITICAL.
    blacklisted_passport.jpg   — otherwise clean, but its passport
                                  number matches a "blacklisted" mock
                                  registry entry -> expect CRITICAL
                                  (modules.risk_engine floors a
                                  registry blacklist hit into the
                                  CRITICAL band regardless of how
                                  clean everything else looks).
"""

import os

import cv2
import numpy as np
from PIL import Image

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _draw_passport_text(img, lines, start_y=60, line_height=55, font_scale=0.85, thickness=2):
    y = start_y
    for line in lines:
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += line_height
    return img


def generate_genuine_passport(path):
    """Clean passport matching mock registry entry X1234567 (valid)."""
    img = np.ones((700, 950, 3), dtype=np.uint8) * 255
    lines = [
        "PASSPORT",
        "Surname: DOE",
        "Given Names: JOHN MICHAEL",
        "Passport No: X1234567",
        "Nationality: EXAMPLIAN",
        "Date of Birth: 15 JAN 1990",
        "Date of Expiry: 20 JAN 2030",
        "Sex: M",
    ]
    _draw_passport_text(img, lines)
    # Single, ordinary JPEG encode — no pasted regions, no EXIF
    # editing signature, nothing to flag.
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def generate_tampered_passport(path):
    """
    Forged passport: pasted-in real face photo with a mismatched
    compression/noise fingerprint, an expiry line rendered in a
    visibly different font/baseline (simulating a hand-edited field)
    showing an expired date, a passport number absent from the
    registry, and a Photoshop EXIF signature.
    """
    from skimage import data as skimage_data  # local import: only needed for fixture generation

    img = np.ones((700, 950, 3), dtype=np.uint8) * 255

    # Explicit (text, y, font_scale, thickness) so nothing can collide
    # with the hand-styled expiry line or the pasted photo region.
    normal_lines = [
        ("PASSPORT", 60, 0.85, 2),
        ("Surname: SMITH", 125, 0.85, 2),
        ("Given Names: ROBERT", 190, 0.85, 2),
        ("Passport No: Z9999999", 255, 0.85, 2),  # not in the mock registry -> Not Found
        ("Nationality: EXAMPLIAN", 320, 0.85, 2),
        ("Date of Birth: 02 JUN 1988", 385, 0.85, 2),
    ]
    for text, y, scale, thickness in normal_lines:
        cv2.putText(img, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Expired date, deliberately rendered in a visibly different font
    # scale/weight than every other field -- simulating a field that
    # was digitally edited after the fact.
    cv2.putText(img, "Date of Expiry: 10 MAR 2019", (40, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "Sex: M", (40, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)

    # A pasted-in *real* face (a genuine detectable face is needed for
    # the photo-substitution check to run at all), generated at high
    # JPEG quality and decoded separately, then pasted into the base
    # image -- exactly the kind of compression/noise discontinuity
    # photo-substitution forensics look for.
    face_rgb = skimage_data.astronaut()
    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    face_crop = cv2.resize(face_bgr, (220, 220))
    ok, encoded_photo = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 98])
    photo_recompressed = cv2.imdecode(encoded_photo, cv2.IMREAD_COLOR)
    img[70:290, 680:900] = photo_recompressed

    # Base image compressed at a distinctly lower quality than the
    # pasted photo region above.
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 55])
    tmp_path = path + ".tmp.jpg"
    with open(tmp_path, "wb") as f:
        f.write(encoded.tobytes())

    # Re-open with PIL to attach a Photoshop EXIF "Software" signature
    # and a modified-after-capture timestamp, then save as the final file.
    pil_img = Image.open(tmp_path)
    exif = Image.Exif()
    exif[0x0131] = "Adobe Photoshop 24.0"   # Software tag
    exif[0x0132] = "2024:03:15 10:00:00"    # DateTime (modified)
    exif[0x9003] = "2018:11:01 08:00:00"    # DateTimeOriginal (capture)
    pil_img.save(path, exif=exif, quality=92)
    os.remove(tmp_path)


def generate_blacklisted_passport(path):
    """Otherwise-clean passport matching mock registry entry B1122334 (blacklisted)."""
    img = np.ones((700, 950, 3), dtype=np.uint8) * 255
    lines = [
        "PASSPORT",
        "Surname: OKAFOR",
        "Given Names: SAMUEL",
        "Passport No: B1122334",
        "Nationality: NIGERIAN",
        "Date of Birth: 22 SEP 1982",
        "Date of Expiry: 01 MAY 2031",
        "Sex: M",
    ]
    _draw_passport_text(img, lines)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    generate_genuine_passport(os.path.join(OUTPUT_DIR, "genuine_passport.jpg"))
    generate_tampered_passport(os.path.join(OUTPUT_DIR, "tampered_passport.jpg"))
    generate_blacklisted_passport(os.path.join(OUTPUT_DIR, "blacklisted_passport.jpg"))
    print("Generated 3 test documents in", OUTPUT_DIR)
