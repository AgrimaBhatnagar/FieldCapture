import unittest
from fieldcapture.ocr import verify_identifier_field
from eval.audit_damage_dataset import audit
from pathlib import Path

class ModelUpgradeTests(unittest.TestCase):
    def test_model_identifier_requires_stronger_confidence(self):
        self.assertFalse(verify_identifier_field('model','AB12345',0.70)['verified'])
        self.assertTrue(verify_identifier_field('model','AB12345',0.80)['verified'])
    def test_serial_identifier_requires_stronger_confidence(self):
        self.assertFalse(verify_identifier_field('serial','AB123456',0.80)['verified'])
        self.assertTrue(verify_identifier_field('serial','AB123456',0.85)['verified'])
    def test_missing_serial_is_not_visible(self):
        self.assertEqual(verify_identifier_field('serial',None,0)['status'],'not_visible')
    def test_manifest_audit_blocks_template(self):
        result=audit(Path(__file__).resolve().parents[1]/'eval'/'damage_manifest.template.jsonl')
        self.assertFalse(result['eligible_for_grouped_validation'])

if __name__=='__main__': unittest.main()
