import unittest
import numpy as np
from fieldcapture.ocr_backend import _as_bgr, _parse_v2, _parse_v3


class PaddleAdapterTests(unittest.TestCase):
    def test_grayscale_image_becomes_bgr(self):
        image = np.zeros((12, 20), dtype=np.uint8)
        result = _as_bgr(image)
        self.assertEqual(result.shape, (12, 20, 3))

    def test_v2_result_is_normalised(self):
        raw = [[[[[1, 2], [11, 2], [11, 8], [1, 8]], ("MODEL123", 0.91)]]]
        rows = _parse_v2(raw)
        self.assertEqual(rows[0][0], "MODEL123")
        self.assertAlmostEqual(rows[0][1], 91.0)
        self.assertEqual(rows[0][2], (1, 2, 10, 6))

    def test_v3_result_is_normalised(self):
        raw = [{"res": {"rec_texts": ["SERIAL123"], "rec_scores": [0.88],
                        "dt_polys": [[[1, 2], [11, 2], [11, 8], [1, 8]]]}}]
        rows = _parse_v3(raw)
        self.assertEqual(rows[0][0], "SERIAL123")
        self.assertAlmostEqual(rows[0][1], 88.0)
        self.assertEqual(rows[0][2], (1, 2, 10, 6))


if __name__ == "__main__":
    unittest.main()
