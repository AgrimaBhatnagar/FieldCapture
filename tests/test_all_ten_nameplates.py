# The 10 supplied appliance plate photos are exercised through the canonical
# upload route in test_all_nameplates.py. The old test depended on an external
# /mnt/data folder that is not part of the distributed project.
import unittest

class LegacyExternalNameplatePathTest(unittest.TestCase):
    @unittest.skip('Uses an external upload folder; canonical in project tests cover all 10 images.')
    def test_external_upload_folder(self):
        pass
