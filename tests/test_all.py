import os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fieldcapture import commands, quality, lineitems, outbox, evaluate


class Commands(unittest.TestCase):
    def test_nameplate(self): self.assertEqual(commands.parse("Capture nameplate."), {"type": "nameplate"})
    def test_damage_full(self):
        self.assertEqual(commands.parse("capture damage kitchen north wall"),
                         {"type": "damage", "room": "kitchen", "surface": "north wall"})
    def test_damage_two_word_room(self):
        self.assertEqual(commands.parse("capture damage living room ceiling")["room"], "living room")
    def test_damage_no_surface(self):
        self.assertIsNone(commands.parse("capture damage kitchen")["surface"])
    def test_note_keeps_case(self):
        self.assertEqual(commands.parse("Note: Subfloor soft by the cabinet")["text"], "Subfloor soft by the cabinet")
    def test_unknown(self): self.assertIsNone(commands.parse("hello there"))


class Quality(unittest.TestCase):
    # Synthetic arrays are UNIT-TEST FIXTURES for the metric maths only. They are not eval data.
    def setUp(self):
        r = np.random.default_rng(0)
        self.sharp = r.random((120, 160)) * 255
    def test_good_passes(self): self.assertTrue(quality.check(self.sharp)["pass"])
    def test_dark_fails(self):
        c = quality.check(self.sharp * 0.1)
        self.assertFalse(c["pass"]); self.assertIn("too dark, turn on a light", c["reasons"])
    def test_flat_is_blurry(self):
        self.assertIn("blurry, hold still", quality.check(np.full((120, 160), 128.0))["reasons"])
    def test_blur_lowers_sharpness(self):
        k = self.sharp.copy()
        for _ in range(4):
            k[1:-1, 1:-1] = (k[:-2, 1:-1] + k[2:, 1:-1] + k[1:-1, :-2] + k[1:-1, 2:] + k[1:-1, 1:-1]) / 5
        self.assertLess(quality.metrics(k)["sharpness"], quality.metrics(self.sharp)["sharpness"] / 5)
    def test_edges_outside_centre_fail_framing(self):
        g = np.zeros((120, 160)); g[:, :30] = np.random.default_rng(1).random((120, 30)) * 255
        self.assertLess(quality.metrics(g)["framing"], 0.8)
    def test_sweep(self):
        rows = [({"luma": 20}, False), ({"luma": 80}, True)]
        self.assertEqual(quality.sweep(rows, "luma", [50])[0]["precision"], 1.0)


def cap(i, t, **f): return {"id": i, "type": t, "fields": {k: {"value": v, "confidence": c} for k, (v, c) in f.items()}}


class Lines(unittest.TestCase):
    def test_ceiling_stain(self):
        L = lineitems.derive([cap("c1", "damage", room=("attic", .9), surface=("ceiling", .9),
                                  damage_type=("water stain", 1), length_ft=(4, 1), height_ft=(3, 1))])
        codes = [l["code"] for l in L]
        self.assertIn("CEI-01", codes); self.assertEqual(L[0]["qty"], 12.0)
        self.assertEqual(codes.count("DOC-01"), 1)
    def test_missing_size_flags_none(self):
        L = lineitems.derive([cap("c1", "damage", room=("kitchen", .9), surface=("north wall", .9), damage_type=("mold", 1))])
        self.assertIsNone(L[0]["qty"]); self.assertLessEqual(L[0]["confidence"], 0.3)
    def test_one_doc_per_room(self):
        d = lambda i: cap(i, "damage", room=("kitchen", .9), surface=("wall", .9), damage_type=("water stain", 1), length_ft=(1, 1), height_ft=(1, 1))
        self.assertEqual([l["code"] for l in lineitems.derive([d("a"), d("b")])].count("DOC-01"), 1)
    def test_catalog_size(self): self.assertTrue(10 <= len(lineitems.CATALOG) <= 15)
    def test_unset_type_makes_nothing(self):
        self.assertEqual(lineitems.derive([cap("c", "damage", room=("x", .9))]), [])


class Outbox(unittest.TestCase):
    def test_offline_then_online_loses_nothing(self):
        o = outbox.Outbox(); [o.put(f"c{i}", {"i": i}) for i in range(5)]
        self.assertEqual(o.flush(lambda b: False, now=0), 0); self.assertEqual(o.pending(), 5)
        self.assertEqual(o.flush(lambda b: True, now=1000), 5); self.assertEqual(o.pending(), 0)
    def test_backoff_blocks_early_retry(self):
        o = outbox.Outbox(); o.put("a", {}); o.flush(lambda b: False, now=0)
        self.assertEqual(o.flush(lambda b: True, now=1), 0)
        self.assertEqual(o.flush(lambda b: True, now=3), 1)
    def test_duplicate_id_ignored(self):
        o = outbox.Outbox(); o.put("a", {}); o.put("a", {}); self.assertEqual(o.pending(), 1)
    def test_send_exception_requeues(self):
        o = outbox.Outbox(); o.put("a", {})
        def boom(b): raise IOError()
        o.flush(boom, now=0); self.assertEqual(o.pending(), 1)
    def test_survives_restart(self):
        p = os.path.join(tempfile.mkdtemp(), "q.db"); o = outbox.Outbox(p); o.put("a", {"x": 1}); o.db.close()
        self.assertEqual(outbox.Outbox(p).pending(), 1)


class Eval(unittest.TestCase):
    # Hand-written rows to test the scoring code. Not a test set, not reported as results.
    def test_scoring(self):
        rows = [{"frame": "a", "kind": "nameplate", "truth": {"serial": "AB-12"}, "pred": {"serial": {"value": "ab12", "confidence": .9}}},
                {"frame": "b", "kind": "nameplate", "truth": {"serial": "ZZ9"}, "pred": {"serial": {"value": "ZZ8", "confidence": .95}}}]
        r = evaluate.score(rows)
        self.assertEqual(r["per_field"]["nameplate.serial"]["accuracy"], 0.5)
        self.assertEqual(r["worst_5"][0]["frame"], "b"); self.assertFalse(evaluate.requirements_met(r))





class Schema(unittest.TestCase):
    def test_valid_record(self):
        from fieldcapture import validate
        c = cap("c1", "damage", room=("attic", .9)); c.update(ts="t", sync="queued")
        for f in c["fields"].values(): f["source"] = "voice"
        rec = {"schema": "fieldcapture.job/0.1", "job_id": "J", "started_at": "t", "session": {"state": "active", "device_mode": "mock_device_kit"}, "captures": [c],
               "line_items": lineitems.derive([c])}
        self.assertEqual(validate.validate(rec), [])
    def test_invalid_record(self):
        from fieldcapture import validate
        self.assertTrue(validate.validate({"schema": "x"}))
if __name__ == "__main__": unittest.main()


class Docs(unittest.TestCase):
    def test_required_docs_exist(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for f in ["README.md", "schema.json", "docs/architecture.md", "docs/product_judgment.md",
                  "docs/market_and_platform.md", "docs/STATUS.md", "docs/research_memo_TEMPLATE.md", "web_prototype.html"]:
            self.assertTrue(os.path.exists(os.path.join(root, f)), f)


if __name__ == "__main__": unittest.main()
