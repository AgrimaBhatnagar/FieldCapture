import unittest
from app import REF_META, ref_fields

class CanonicalReferenceDimensionTests(unittest.TestCase):
    def test_all_reference_records_keep_original_supplied_measurements(self):
        self.assertEqual(len(REF_META['records']), 40)
        for rec in REF_META['records']:
            fields = ref_fields(rec)
            for key in ('length_ft', 'width_ft'):
                if key in fields:
                    self.assertIsNotNone(fields[key]['value'], (rec['id'], key))
                    self.assertGreaterEqual(fields[key]['confidence'], 0.90)
                    self.assertLessEqual(fields[key]['confidence'], 1.00)
                    self.assertEqual(fields[key]['source'], 'reference_annotation')

    def test_nameplates_do_not_receive_invented_dimensions(self):
        for rec in REF_META['records']:
            if rec['id'].startswith('nameplate_'):
                fields = ref_fields(rec)
                self.assertNotIn('length_ft', fields)
                self.assertNotIn('width_ft', fields)

    def test_dimension_ranges_are_preserved(self):
        by_id = {r['id']: r for r in REF_META['records']}
        fields = ref_fields(by_id['inspection_10'])
        self.assertEqual(fields['length_ft']['value'], [20, 25])
        self.assertEqual(fields['width_ft']['value'], [15, 20])

if __name__ == '__main__':
    unittest.main()
