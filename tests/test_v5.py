import json
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fieldcapture import session, commands, quality, lineitems, outbox, evaluate, validate, pipeline


class CompleteAssessmentTests(unittest.TestCase):
    def test_session_lifecycle_and_disconnect_recovery(self):
        ctrl = session.SessionController(job_id="job-1", started_at="2026-11-23T16:00:00Z")
        self.assertEqual(ctrl.state, "idle")
        self.assertTrue(ctrl.connected)
        ctrl.register_and_grant_permissions(audio_available=False)
        self.assertIn("phone microphone", ctrl.audio_fallback_reason)
        self.assertIn("permissions_granted", ctrl.events)

        ctrl.start()
        self.assertEqual(ctrl.state, "active")
        ctrl.pause()
        self.assertEqual(ctrl.state, "paused")
        ctrl.resume()
        self.assertEqual(ctrl.state, "active")

        ctrl.simulate_disconnect()
        self.assertFalse(ctrl.connected)
        ctrl.recover_after_disconnect()
        self.assertTrue(ctrl.connected)
        self.assertEqual(ctrl.state, "active")
        self.assertEqual(ctrl.drop_recoveries, 1)

        ctrl.stop()
        self.assertEqual(ctrl.state, "stopped")

    def test_commands_and_note_extraction(self):
        self.assertIsNone(commands.parse(""))
        self.assertIsNone(commands.parse(None))
        self.assertEqual(commands.parse("capture nameplate."), {"type": "nameplate"})
        self.assertEqual(
            commands.parse("capture damage, kitchen, north wall"),
            {"type": "damage", "room": "kitchen", "surface": "north wall"},
        )
        self.assertEqual(
            commands.parse("note: subfloor soft by the cabinet"),
            {"type": "note", "text": "subfloor soft by the cabinet"},
        )

        nf = commands.parse_note_fields("subfloor soft by the cabinet in kitchen")
        self.assertEqual(nf["room"]["value"], "kitchen")
        self.assertEqual(nf["room"]["confidence"], 0.9)
        self.assertEqual(nf["surface"]["value"], "subfloor")
        self.assertEqual(nf["condition"]["value"], "soft")

    def test_quality_load_check_and_sweep(self):
        rng = np.random.default_rng(7)
        good = rng.random((120, 160)) * 200 + 30
        dark = rng.random((120, 160)) * 5
        self.assertTrue(quality.check(good)["pass"])
        bad = quality.check(dark)
        self.assertFalse(bad["pass"])
        self.assertIn("too dark", bad["retake_prompt"])

        rows = [
            (quality.metrics(good), True),
            (quality.metrics(dark), False),
        ]
        sweep_res = quality.sweep(rows, "luma", [10.0, 25.0])
        self.assertEqual(len(sweep_res), 2)
        self.assertEqual(sweep_res[1]["cutoff"], 25.0)
        self.assertEqual(sweep_res[1]["precision"], 1.0)
        self.assertEqual(sweep_res[1]["recall"], 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, "sample.png")
            Image.fromarray(np.clip(good, 0, 255).astype(np.uint8), mode="L").save(img_path)
            loaded = quality.load_grayscale(img_path, size=(160, 120))
            self.assertEqual(loaded.shape, (120, 160))

    def test_lineitems_and_depth_free_estimation(self):
        w_ft, h_ft, conf = lineitems.estimate_size_without_depth(1280, 720, 320, 240, assumed_dist_ft=4.0)
        self.assertGreater(w_ft, 0)
        self.assertGreater(h_ft, 0)
        self.assertEqual(conf, 0.45)
        self.assertEqual(lineitems.estimate_size_without_depth(0, 720, 100, 100), (None, None, 0.0))

        captures = [
            {"id": "n1", "type": "nameplate", "fields": {}},
            {
                "id": "d1",
                "type": "damage",
                "fields": {
                    "room": {"value": "bathroom", "confidence": 0.9, "source": "voice"},
                    "surface": {"value": "ceiling", "confidence": 0.85, "source": "voice"},
                    "damage_type": {"value": "mold", "confidence": 0.88, "source": "model"},
                    "length_ft": {"value": 4.0, "confidence": 0.5, "source": "model"},
                    "height_ft": {"value": 3.0, "confidence": 0.5, "source": "model"},
                },
            },
        ]
        items = lineitems.derive(captures)
        codes = {it["code"] for it in items}
        self.assertTrue({"WH-01", "CEI-01", "DRY-02", "MLD-01", "DOC-01"}.issubset(codes))

    def test_pipeline_outbox_validation_and_evaluate(self):
        ctrl = session.SessionController(job_id="job-e2e", started_at="2026-11-23T16:10:00Z")
        ctrl.start()
        rng = np.random.default_rng(9)
        sharp_gray = rng.random((120, 160)) * 200 + 30
        job = pipeline.build_job_record(
            "job-e2e",
            "2026-11-23T16:10:00Z",
            ctrl,
            [
                {
                    "id": "c-1",
                    "type": "nameplate",
                    "gray": sharp_gray,
                    "fields": {
                        "make": {"value": "Rheem", "confidence": 0.92, "source": "model"},
                        "model": {"value": "PROG50-38N", "confidence": 0.89, "source": "model"},
                    },
                },
                {
                    "id": "c-2",
                    "type": "damage",
                    "gray": sharp_gray,
                    "fields": {
                        "room": {"value": "kitchen", "confidence": 0.9, "source": "voice"},
                        "surface": {"value": "north wall", "confidence": 0.85, "source": "voice"},
                        "damage_type": {"value": "water stain", "confidence": 0.88, "source": "model"},
                        "length_ft": {"value": 3.0, "confidence": 0.5, "source": "model"},
                        "height_ft": {"value": 2.0, "confidence": 0.5, "source": "model"},
                    },
                },
                {
                    "id": "c-3",
                    "type": "note",
                    "note_text": "subfloor soft by the cabinet",
                },
            ],
        )
        self.assertEqual(validate.validate(job), [])
        self.assertNotEqual(validate.validate({"schema": "wrong"}), [])

        with tempfile.TemporaryDirectory() as tmp:
            with outbox.Outbox(os.path.join(tmp, "outbox.db")) as ob:
                ob.put("c-1", job["captures"][0], now=10.0)
                ob.put("c-1", job["captures"][0], now=10.0)
                self.assertEqual(ob.pending(), 1)
                # Failing send triggers exponential backoff
                self.assertEqual(ob.flush(lambda x: False, now=11.0, base=2.0), 0)
                self.assertEqual(ob.pending(), 1)
                # Before next_at (13.0), not flushed
                self.assertEqual(ob.flush(lambda x: True, now=12.0, base=2.0), 0)
                # At >= 13.0, flushed
                self.assertEqual(ob.flush(lambda x: True, now=14.0, base=2.0), 1)
                self.assertEqual(ob.pending(), 0)
                self.assertEqual(ob.avg_sync_latency_sec(), 4.0)

            eval_rows = [
                {
                    "frame": "f1",
                    "kind": "nameplate",
                    "lowlight": True,
                    "truth": {"make": "Rheem"},
                    "pred": {"make": {"value": "Rheem", "confidence": 0.9}},
                },
                {
                    "frame": "f2",
                    "kind": "nameplate",
                    "lowlight": False,
                    "truth": {"model": "ABC123"},
                    "pred": {"model": {"value": "WRONG", "confidence": 0.75}},
                },
            ]
            rep = evaluate.score(eval_rows)
            self.assertEqual(rep["frames"], 2)
            self.assertEqual(rep["nameplates"], 2)
            self.assertEqual(rep["lowlight"], 1)
            self.assertFalse(evaluate.requirements_met(rep))

            jsonl_path = os.path.join(tmp, "frames.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for r in eval_rows:
                    f.write(json.dumps(r) + "\n")

            cli_res = subprocess.run(
                [sys.executable, "-m", "fieldcapture.evaluate", jsonl_path],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cli_res.returncode, 0)
            cli_rep = json.loads(cli_res.stdout)
            self.assertEqual(cli_rep["frames"], 2)


if __name__ == "__main__":
    unittest.main()
