import json
import unittest
from pathlib import Path

import app

ROOT = Path(__file__).resolve().parents[1]


class OCROverlayAndReferenceConfidenceTests(unittest.TestCase):
    def test_reference_confidence_varies_within_requested_band(self):
        for rec in app.REF_META["records"]:
            fields = app.ref_fields(rec)
            scores = [v["confidence"] for k, v in fields.items()
                      if not k.startswith("_") and isinstance(v, dict) and "confidence" in v]
            self.assertTrue(scores)
            self.assertTrue(all(0.90 <= score <= 1.00 for score in scores))
        sample = app.ref_fields(app.REF_BY_ID["inspection_01"])
        self.assertGreater(len({v["confidence"] for k, v in sample.items()
                                if not k.startswith("_") and isinstance(v, dict)}), 1)

    def test_reference_ocr_boxes_are_preserved_for_capture_overlay(self):
        fields = app.ref_fields(app.REF_BY_ID["inspection_01"])
        ocr_boxes = [b for b in fields["_bounding_boxes"] if b.get("kind") == "ocr"]
        self.assertTrue(ocr_boxes)
        self.assertTrue(all(len(b["box"]) == 4 for b in ocr_boxes))
        html = (ROOT / "web" / "app.html").read_text(encoding="utf-8")
        self.assertIn("function drawStageOCR(c)", html)
        self.assertIn("drawStageOCR(r.capture)", html)
        self.assertIn("class=\"ocrBox\"", html)


if __name__ == "__main__":
    unittest.main()
