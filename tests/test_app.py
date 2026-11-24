import json, os, sys, threading, unittest, urllib.request, base64
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app
from http.server import ThreadingHTTPServer


class AppServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), app.H); cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
    @classmethod
    def tearDownClass(cls): cls.srv.shutdown()
    def post(self, u, b):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{u}", json.dumps(b).encode(), {"Content-Type": "application/json"}); return json.load(urllib.request.urlopen(r, timeout=60))
    def get(self, u): return json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}{u}", timeout=60))
    def setUp(self): self.post("/api/reset", {})
    def test_health_and_page(self):
        self.assertTrue(self.get("/api/health")["ok"])
        self.assertIn(b"FieldCapture", urllib.request.urlopen(f"http://127.0.0.1:{self.port}/").read())
    def test_note_from_another_thread_and_valid_job(self):
        r = self.post("/api/capture", {"command": "note: subfloor soft by the cabinet in the kitchen"}); self.assertTrue(r["accepted"])
        j = self.get("/api/job"); self.assertEqual(j["record"]["schema"], "fieldcapture.job/0.1"); self.assertEqual(len(j["captures"]), 1)
    def test_dark_frame_is_rejected_not_stored(self):
        ok, buf = cv2.imencode(".jpg", np.full((300, 300, 3), 8, np.uint8)); img = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        r = self.post("/api/capture", {"command": "capture damage kitchen wall", "image": img})
        self.assertFalse(r["accepted"]); self.assertIn("too dark", r["retake_prompt"]); self.assertEqual(len(self.get("/api/job")["captures"]), 0)
    def test_offline_queues_then_syncs(self):
        self.post("/api/signal", {"online": False}); self.post("/api/capture", {"command": "note: offline one"})
        self.assertEqual(self.get("/api/job")["pending"], 1)
        self.post("/api/signal", {"online": True}); self.assertEqual(self.get("/api/job")["pending"], 0)
    def test_bad_command_and_missing_frame(self):
        self.assertFalse(self.post("/api/capture", {"command": "hello"})["accepted"]); self.assertFalse(self.post("/api/capture", {"command": "capture nameplate"})["accepted"])
    def test_edit_sets_tech_source(self):
        self.post("/api/capture", {"command": "note: x"}); cid = self.get("/api/job")["captures"][0]["id"]
        self.assertTrue(self.post("/api/edit", {"id": cid, "field": "room", "value": "attic"})["ok"])
        self.assertEqual(self.get("/api/job")["captures"][0]["fields"]["room"]["source"], "tech")
    def test_uploaded_reference_photo_uses_user_supplied_record(self):
        rec = app.REF_BY_ID["extra_02"]
        img = cv2.imread(str(app.REF_DIR / rec["filename"]))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        data = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        r = self.post("/api/capture", {"command": "capture damage living room wall", "image": data})
        self.assertTrue(r["accepted"])
        self.assertEqual(r["reference_match"], "extra_02")
        fields = r["capture"]["fields"]
        self.assertEqual(fields["room"]["value"], "Living room")
        self.assertEqual(fields["surface"]["value"], "Wall")
        self.assertEqual(fields["damage_type"]["value"], "dampness_and_vandalism")
        self.assertEqual(fields["damage_presence"]["value"], True)
        self.assertEqual(fields["length_ft"]["value"], [12, 14])
        self.assertEqual(fields["width_ft"]["value"], 2.5)
        self.assertGreaterEqual(fields["length_ft"]["confidence"], 0.90)
        self.assertLessEqual(fields["length_ft"]["confidence"], 1.00)


if __name__ == "__main__": unittest.main()
