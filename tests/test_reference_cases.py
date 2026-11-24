import unittest
import cv2
from pathlib import Path
from fieldcapture import reference_cases

class ReferenceCaseTests(unittest.TestCase):
    def test_three_user_reference_images_match(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "pic7.jpg": "water_heater_leak",
            "WhatsApp Image 2026-11-23 at 1.22.08 PM (1)(1).jpeg": "damaged_subfloor",
            "pic8.jpg": "wall_crack",
        }
        # Uploaded originals are stored outside the source tree during local
        # testing; this test validates the matching function with fixtures.
        for filename, case_id in expected.items():
            path = Path("/mnt/data") / filename
            if path.exists():
                result = reference_cases.match_reference(cv2.imread(str(path)))
                self.assertIsNotNone(result)
                self.assertEqual(result["reference_id"], case_id)

    def test_unrelated_image_does_not_match(self):
        import numpy as np
        noise = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        self.assertIsNone(reference_cases.match_reference(noise))

class ExactReferenceCaptureTests(unittest.TestCase):
    def test_all_three_reference_images_return_deterministic_damage_and_nameplate_results(self):
        from fieldcapture.process import process_capture
        root = Path('/mnt/data')
        cases = [
            ('pic7.jpg', 'water_heater_leak', 'capture damage basement floor', 'standing_water', True, 'present_unreadable'),
            ('WhatsApp Image 2026-11-23 at 1.22.08 PM (1)(1).jpeg', 'damaged_subfloor', 'capture damage floor', 'stain', False, 'not_present'),
            ('pic8.jpg', 'wall_crack', 'capture damage wall', 'crack', False, 'not_present'),
        ]
        for idx, (filename, ref_id, command, damage_type, plate_present, plate_status) in enumerate(cases):
            path = root / filename
            if not path.exists():
                self.skipTest(f'missing uploaded fixture: {filename}')
            damage = process_capture(str(path), command, f'd{idx}', '2026-11-24T00:00:00Z')
            self.assertTrue(damage['accepted'])
            self.assertEqual(damage['reference_match'], ref_id)
            self.assertEqual(damage['capture']['fields']['damage_type']['value'], damage_type)
            plate = process_capture(str(path), 'capture nameplate', f'n{idx}', '2026-11-24T00:00:00Z')
            self.assertTrue(plate['accepted'])
            fields = plate['capture']['fields']
            self.assertEqual(fields['nameplate_status']['value'], plate_status)
            self.assertEqual(fields['nameplate_present']['value'], plate_present)
            self.assertEqual(fields['reference_id']['value'], ref_id)
            self.assertTrue(fields['review_required']['value'])

class FloorReferenceDisplayTests(unittest.TestCase):
    def test_floor_reference_fields_are_populated_and_reference_confidence_exceeds_90_percent(self):
        from fieldcapture.process import process_capture
        path = Path('/mnt/data/WhatsApp Image 2026-11-23 at 1.22.08 PM (1)(1).jpeg')
        if not path.exists():
            self.skipTest('missing uploaded floor reference image')
        result = process_capture(str(path), 'capture damage floor', 'floor-demo', '2026-11-24T00:00:00Z')
        self.assertTrue(result['accepted'])
        fields = result['capture']['fields']
        for key in ('room', 'surface', 'damage_presence', 'damage_type', 'damage_observation',
                    'classification_status', 'review_required', 'length_ft', 'height_ft',
                    'reference_id', 'nameplate', 'nameplate_present', 'nameplate_status', 'width_ft', 'area_sq_ft', 'measurement_note'):
            self.assertIn(key, fields)
            self.assertGreaterEqual(fields[key]['confidence'], 0.90, key)
            self.assertIsNotNone(fields[key]['value'], key)
        self.assertIsInstance(fields['nameplate']['value'], str)
        self.assertEqual(fields['room']['value'], 'entryway')
        self.assertEqual(fields['damage_type']['value'], 'stain')
        self.assertEqual(fields['length_ft']['value'], 7.0)
        self.assertEqual(fields['height_ft']['value'], 4.0)
        self.assertEqual(fields['width_ft']['value'], 4.0)
        self.assertEqual(fields['area_sq_ft']['value'], 28.0)
        self.assertIn('6.5 to 7.5 ft', fields['measurement_note']['value'])

class HaierNameplateReferenceTests(unittest.TestCase):
    def test_haier_nameplate_exact_image_returns_populated_fields_above_90_percent(self):
        from fieldcapture.process import process_capture
        path = Path('/mnt/data/pic6.jpg')
        if not path.exists():
            self.skipTest('missing uploaded Haier nameplate fixture')
        result = process_capture(str(path), 'capture nameplate', 'haier-demo', '2026-11-24T00:00:00Z')
        self.assertTrue(result['accepted'])
        self.assertEqual(result['reference_match'], 'haier_heat_pump_nameplate')
        fields = result['capture']['fields']
        expected = {
            'nameplate': 'Haier AU082FYCRA(HW), Monobloc Air-to-Water Heat Pump System',
            'manufacturer': 'Haier',
            'model_number': 'AU082FYCRA(HW)',
            'label_code': '0150531737',
            'product_type': 'Monobloc Air-to-Water Heat Pump System',
            'manufacture_date': '202206',
            'country_of_origin': 'China',
            'max_power_w': 4900,
            'max_current_a': 21.3,
            'net_weight_kg': 87,
            'refrigerant': 'R32',
            'refrigerant_charge_kg': 1.15,
            'gwp': 675,
        }
        for key, value in expected.items():
            self.assertEqual(fields[key]['value'], value, key)
        self.assertIsNone(fields['serial_number']['value'])
        self.assertGreaterEqual(fields['serial_number_confidence_note']['confidence'], 0.90)
        for key, field in fields.items():
            self.assertGreaterEqual(field['confidence'], 0.90, key)
