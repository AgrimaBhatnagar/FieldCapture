import unittest
from fieldcapture import process, validate

class ValidationMetadataTests(unittest.TestCase):
    def test_reference_bounding_boxes_do_not_break_job_validation(self):
        capture = {
            "id": "c1",
            "type": "damage",
            "ts": "2026-11-24T14:00:00Z",
            "sync": "queued",
            "quality": None,
            "fields": {
                "room": {"value": "entryway", "confidence": 0.98, "source": "reference_annotation"},
                "surface": {"value": "floor", "confidence": 0.98, "source": "reference_annotation"},
                "damage_type": {"value": "water_intrusion", "confidence": 0.98, "source": "reference_annotation"},
                "length_ft": {"value": "6.5 to 7.5", "confidence": 0.98, "source": "reference_annotation"},
                "width_ft": {"value": "4", "confidence": 0.98, "source": "reference_annotation"},
                "_bounding_boxes": [
                    {"kind": "ocr", "text": "TOUGH QUIET", "box": [412, 118, 488, 225]}
                ],
            },
        }
        record = process.build_record(
            "JOB-TEST", "2026-11-24T14:00:00Z", [capture],
            device_mode="browser_webcam"
        )
        self.assertEqual(record["captures"][0]["fields"]["_bounding_boxes"][0]["text"], "TOUGH QUIET")
        self.assertEqual(validate.validate(record), [])

if __name__ == "__main__":
    unittest.main()
