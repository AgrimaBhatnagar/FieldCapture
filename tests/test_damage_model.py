import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from fieldcapture.extract import extract_damage

ROOT = Path(__file__).resolve().parents[1]


class TrainedDamagePresenceModel(unittest.TestCase):
    def test_checkpoint_is_packaged(self):
        self.assertTrue((ROOT / "models" / "damage_presence_rf.pkl").is_file())

    def test_trained_model_runs_and_marks_subtype_for_review(self):
        manifest = json.loads((ROOT / "eval" / "reference_damage" / "manifest.json").read_text())
        row = next(x for x in manifest if x["label"] != "none")
        image = cv2.imread(str(ROOT / "eval" / "reference_damage" / row["file"]))
        result = extract_damage(image)
        self.assertIn(result["damage_presence"]["source"], ["random-forest-presence"])
        self.assertEqual(result["damage_presence"]["value"], "damage")
        self.assertEqual(result["status"], "review_required")
        self.assertTrue(result["review_required"])

    def test_low_information_image_abstains(self):
        result = extract_damage(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["damage_presence"]["value"])


if __name__ == "__main__":
    unittest.main()
