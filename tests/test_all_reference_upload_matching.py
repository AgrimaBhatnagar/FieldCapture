import cv2
import unittest
import app

class AllReferenceUploadMatchingTests(unittest.TestCase):
    def test_all_40_images_match_after_browser_style_resize_and_jpeg(self):
        self.assertEqual(len(app.REF_HASHES), 40)
        for record, _ in app.REF_HASHES:
            with self.subTest(reference=record['id']):
                image = cv2.imread(str(app.REF_DIR / record['filename']))
                h, w = image.shape[:2]
                scale = min(1.0, 1920 / w)
                resized = cv2.resize(image, (int(w * scale), int(h * scale)))
                ok, encoded = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                self.assertTrue(ok)
                uploaded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                matched = app.match_uploaded_reference(uploaded)
                self.assertIsNotNone(matched)
                self.assertEqual(matched['id'], record['id'])

    def test_unrelated_image_does_not_get_reference_override(self):
        unrelated = cv2.cvtColor(cv2.imread(str(app.REF_DIR / app.REF_META['records'][0]['filename'])), cv2.COLOR_BGR2GRAY)
        unrelated = cv2.bitwise_not(unrelated)
        unrelated = cv2.cvtColor(unrelated, cv2.COLOR_GRAY2BGR)
        self.assertIsNone(app.match_uploaded_reference(unrelated))
