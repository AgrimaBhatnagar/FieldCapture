import base64, json, unittest
from pathlib import Path
import app

ROOT = Path(__file__).resolve().parents[1]

class UIDisplayAndOCRTests(unittest.TestCase):
    def test_internal_reference_metadata_is_hidden_from_capture_fields_ui(self):
        html = (ROOT / "web" / "app.html").read_text(encoding="utf-8")
        self.assertIn('["reference_id","reference_override","annotation_source"].includes(k)', html)

    def test_uploaded_reference_image_returns_ocr_boxes_before_capture(self):
        image = (ROOT / "reference_images" / "pic1_3.jpg").read_bytes()
        data_url = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
        result = app.api_ocr_preview({"image": data_url})
        self.assertEqual(result["source"], "reference_annotation")
        self.assertEqual(result["reference_match"], "inspection_01")
        self.assertTrue(any(box.get("text") == "Vector" and box.get("kind") == "ocr" for box in result["boxes"]))

    def test_new_image_ocr_preview_endpoint_is_wired_to_upload_handler(self):
        html = (ROOT / "web" / "app.html").read_text(encoding="utf-8")
        self.assertIn('await previewOCR()', html)
        self.assertIn('post("/api/ocr_preview"', html)
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('self.path == "/api/ocr_preview"', app_source)

if __name__ == "__main__": unittest.main()
