import base64
import cv2
import unittest
import app


class AllTenNameplateTests(unittest.TestCase):
    def test_all_ten_reference_images_match_and_return_populated_fields(self):
        expected = {i: app.REF_BY_ID[f'nameplate_{i:02d}']['fields']['manufacturer'] for i in range(1, 11)}
        for i, brand in expected.items():
            with self.subTest(image=i):
                rec = app.REF_BY_ID[f'nameplate_{i:02d}']
                image = cv2.imread(str(app.REF_DIR / rec['filename']))
                ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                self.assertTrue(ok)
                data = 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode()
                result = app.api_capture({'command': 'capture nameplate', 'image': data})
                self.assertTrue(result['accepted'])
                self.assertEqual(result['reference_match'], rec['id'])
                fields = result['capture']['fields']
                self.assertEqual(fields['manufacturer']['value'], brand)
                self.assertGreaterEqual(fields['manufacturer']['confidence'], .90)
                self.assertTrue(all(v['confidence'] >= .90 for v in fields.values() if isinstance(v, dict) and 'confidence' in v))
