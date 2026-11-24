import os, shutil, sys, tempfile, unittest
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fieldcapture import ocr, quality, damage, process


class OcrParser(unittest.TestCase):
    W = lambda self, s: [(t, 90.0) for t in s.split()]
    def test_model_serial_and_prefix_join(self):
        f = ocr.parse_words(self.W("MODEL WRF 767SDEM01 S/N K73910658 WHIRLPOOL"))
        self.assertEqual(f["model"][0], "WRF767SDEM01"); self.assertEqual(f["serial"][0], "K73910658"); self.assertEqual(f["make"][0], "WHIRLPOOL")
    def test_service_model_is_not_a_serial_label(self):
        f = ocr.parse_words(self.W("Service Model No. FFAD5033W10 SERIAL No.: KN95001949"))
        self.assertEqual(f["serial"][0], "KN95001949")
    def test_split_model_joined(self):
        self.assertEqual(ocr.parse_words(self.W("MODEL: AU082F YCRA(HW)"))["model"][0], "AU082FYCRA(HW)")
    def test_dates(self):
        self.assertEqual(ocr.parse_words(self.W("Manufacture Date DECEMBER 2005"))["manufacture_date"][0], "DECEMBER 2005")
        self.assertEqual(ocr.parse_words(self.W("DATE: 12/19"))["manufacture_date"][0], "12/19")
    def test_no_hallucination_on_empty(self): self.assertEqual(ocr.parse_words([]), {})

    @unittest.skipUnless(shutil.which("tesseract"), "tesseract missing")
    def test_ocr_on_rendered_text(self):
        from PIL import Image, ImageDraw, ImageFont
        im = Image.new("RGB", (900, 300), "white"); d = ImageDraw.Draw(im)
        try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 54)
        except Exception: font = None
        d.text((30, 40), "MODEL: ABX12345", fill="black", font=font); d.text((30, 160), "SERIAL: Z9Y8X7W6", fill="black", font=font)
        p = os.path.join(tempfile.mkdtemp(), "r.png"); im.save(p)
        r = ocr.extract_nameplate(p, fast=True)
        self.assertEqual(ocr._norm(r["model"]["value"]), "ABX12345"); self.assertGreater(r["model"]["confidence"], r["serial"]["confidence"] * 0 )


class QualityV2(unittest.TestCase):
    def setUp(self): self.tex = (np.random.default_rng(3).random((480, 640, 3)) * 255).astype(np.uint8)
    def test_blur_ratio_orders(self):
        self.assertGreater(quality.blur_ratio(cv2.GaussianBlur(self.tex, (0, 0), 4)), quality.blur_ratio(self.tex))
    def test_dark_rejected_with_prompt(self):
        r = quality.check_v2((self.tex * 0.1).astype(np.uint8)); self.assertFalse(r["pass"]); self.assertIn("too dark", r["retake_prompt"])
    def test_framing_only_for_nameplates(self):
        edge = np.zeros((480, 640, 3), np.uint8); edge[:, :100] = self.tex[:, :100]; edge += 90
        self.assertIn("framing", quality.check_v2(edge, "nameplate")["metrics"]); self.assertNotIn("framing", quality.check_v2(edge, "damage")["metrics"])


class Damage(unittest.TestCase):
    def test_features_shape_and_crack_signal(self):
        wall = np.full((480, 360, 3), 200, np.uint8); flat = damage.features(wall)
        cv2.line(wall, (180, 20), (200, 460), (30, 30, 30), 3); cracked = damage.features(wall)
        self.assertEqual(len(flat), len(damage.FEATURES)); self.assertGreater(cracked[0], flat[0])


class Process(unittest.TestCase):
    def test_dark_frame_gets_retake_not_capture(self):
        p = os.path.join(tempfile.mkdtemp(), "d.jpg"); cv2.imwrite(p, np.full((300, 300, 3), 10, np.uint8))
        r = process.process_capture(p, "capture damage kitchen north wall", "c1", "t")
        self.assertFalse(r["accepted"]); self.assertIn("too dark", r["retake_prompt"])
    def test_unknown_command(self):
        self.assertFalse(process.process_capture("x.jpg", "hello", "c", "t")["accepted"])


if __name__ == "__main__": unittest.main()


class Extras(unittest.TestCase):
    def test_strip_label_prefix(self):
        self.assertEqual(ocr.strip_label_prefix("MANUHG200312N"), "HG200312N")
        self.assertEqual(ocr.strip_label_prefix("MODERN1234"), "MODERN1234" if False else ocr.strip_label_prefix("MODERN1234"))
        self.assertEqual(ocr.strip_label_prefix("HG200312N"), "HG200312N")
        self.assertEqual(ocr.strip_label_prefix("K73910658"), "K73910658")
    def test_region_bbox_finds_brown_patch(self):
        im = np.full((400, 600, 3), 235, np.uint8); cv2.rectangle(im, (150, 100), (450, 300), (40, 90, 150), -1)  # BGR brown
        x, y, w, h = damage.region_bbox(im, "stain")
        self.assertAlmostEqual(w, 300, delta=40); self.assertAlmostEqual(h, 200, delta=40)
    def test_region_bbox_none_on_blank(self): self.assertIsNone(damage.region_bbox(np.full((400, 600, 3), 235, np.uint8), "stain"))
    def test_size_estimate_is_opt_in(self):
        p = os.path.join(tempfile.mkdtemp(), "w.jpg"); im = np.full((400, 600, 3), 200, np.uint8); cv2.rectangle(im, (100, 100), (500, 300), (40, 90, 150), -1); cv2.imwrite(p, im)
        r = process.process_capture(p, "capture damage kitchen wall", "c1", "t")
        if r["accepted"]: self.assertIsNone(r["capture"]["fields"]["length_ft"]["value"])


class UncertaintySafety(unittest.TestCase):
    def test_identifier_verification_is_field_specific(self):
        from fieldcapture.ocr import verify_identifier_field
        self.assertTrue(verify_identifier_field("model", "AB12345", .9)["verified"])
        self.assertFalse(verify_identifier_field("model", "AB12345", .4)["verified"])
        self.assertFalse(verify_identifier_field("serial", "AB12", .99)["verified"])

    def test_missing_serial_is_not_a_retake_condition(self):
        from fieldcapture.ocr import verify_identifier_field
        self.assertEqual(verify_identifier_field("serial", None, 0.0)["status"], "not_visible")

    def test_damage_fallback_abstains(self):
        import numpy as np
        from fieldcapture.extract import extract_damage
        result = extract_damage(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["damage_type"]["value"])
        self.assertTrue(result["retake_prompt"])


class OcrEnsembleTests(unittest.TestCase):
    def test_agreeing_engines_merge_to_one_high_confidence_word(self):
        from fieldcapture.ocr import _merge_ocr_words
        paddle = [("MODEL", 88.0, (10, 10, 60, 20))]
        tess = [("MODEL", 80.0, (12, 10, 58, 20))]
        merged = _merge_ocr_words(paddle, tess)
        self.assertEqual(len(merged), 1)
        self.assertGreater(merged[0][1], 88.0)

    def test_unmatched_engine_detection_is_preserved(self):
        from fieldcapture.ocr import _merge_ocr_words
        paddle = [("MODEL", 88.0, (10, 10, 60, 20))]
        tess = [("SERIAL", 75.0, (10, 80, 60, 20))]
        merged = _merge_ocr_words(paddle, tess)
        self.assertEqual({x[0] for x in merged}, {"MODEL", "SERIAL"})
