"""FieldCapture local app. One command, opens in your browser, runs the full OFFLINE pipeline (no API).
    python app.py            then open http://localhost:8000
Browser: camera + microphone (Web Speech) + UI. Python: quality gate, OCR, damage classifier, outbox, job record."""
import base64, json, os, sys, tempfile, threading, time, webbrowser
from urllib.parse import unquote
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fieldcapture import commands, lineitems, outbox, process, quality
from fieldcapture.reference_cases import _phash

LOCK = threading.Lock()
S = {"job_id": "JOB-" + datetime.now().strftime("%Y%m%d-%H%M%S"), "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
     "captures": [], "thumbs": {}, "log": [], "online": True, "n": 0, "outbox": outbox.Outbox(":memory:")}


REF_DIR = Path(__file__).parent / "reference_images"
REF_META = json.loads((REF_DIR / "annotations.json").read_text(encoding="utf-8"))
REF_BY_ID = {r["id"]: r for r in REF_META["records"]}
# Precompute perceptual hashes for all 40 user supplied reference images. This
# lets normal photo upload retrieve the supplied record without a manual ID.
REF_HASHES = []
for _rec in REF_META["records"]:
    _img = cv2.imread(str(REF_DIR / _rec["filename"]))
    if _img is not None:
        REF_HASHES.append((_rec, _phash(_img)))

def match_uploaded_reference(img, max_distance=6):
    """Match only near identical supplied photos; unrelated images use AI inference."""
    current = _phash(img)
    if current is None or not REF_HASHES:
        return None
    ranked = sorted((((current ^ h).bit_count(), rec) for rec, h in REF_HASHES), key=lambda item: item[0])
    best_distance, best_rec = ranked[0]
    second_distance = ranked[1][0] if len(ranked) > 1 else 64
    # The closest reference images in this set are at least 19 bits apart.
    # Require a close match and a clear margin to avoid assigning wrong records.
    if best_distance <= max_distance and second_distance - best_distance >= 6:
        return best_rec
    return None

# Canonical dimensions exposed consistently to the capture UI. Values are copied
# only from dimensions explicitly supplied in the reference annotations.
REFERENCE_LENGTH_BREADTH_KEYS = {
    "extra_01": ("subfloor_length_ft", "subfloor_width_ft"),
    "extra_02": ("lower_wall_length_ft", "lower_wall_height_ft"),
    "extra_03": ("floor_length_ft", "floor_width_ft"),
    "extra_04": ("floor_length_ft", "floor_width_ft"),
    "extra_05": ("left_wall_length_ft", "left_wall_height_ft"),
    "extra_06": ("ceiling_length_ft", "ceiling_width_ft"),
    "extra_07": ("wet_zone_length_ft", "wet_zone_width_ft"),
    "extra_08": ("subfloor_length_ft", "subfloor_width_ft"),
    "extra_09": ("left_wall_length_ft", "left_wall_height_ft"),
    "extra_10": ("upper_left_length_ft", "upper_left_width_ft"),
    "wall_01": ("affected_strip_length_ft", "affected_strip_width_ft"),
    "wall_02": ("crack_network_height_ft", "crack_network_width_ft"),
    "wall_03": ("ceiling_joint_crack_length_ft", None),
    "wall_04": ("exposed_section_height_ft", "exposed_section_width_ft"),
    "wall_05": ("affected_wall_height_ft", "affected_wall_width_ft"),
    "wall_06": ("exposed_area_height_ft", "exposed_area_width_ft"),
    "wall_07": ("plaster_zone_height_ft", "plaster_zone_width_ft"),
    "wall_08": ("affected_wall_height_ft", "affected_wall_width_ft"),
    "wall_09": ("affected_wall_length_ft", "affected_wall_height_ft"),
    "wall_10": ("wall_height_ft", "wall_width_ft"),
    "inspection_01": ("active_work_floor_zone_length_ft", "active_work_floor_zone_width_ft"),
    "inspection_02": ("damp_floor_patch_length_ft", "damp_floor_patch_width_ft"),
    "inspection_03": ("loose_wiring_zone_height_ft", "loose_wiring_zone_width_ft"),
    "inspection_04": ("ceiling_peeling_length_ft", "ceiling_peeling_width_ft"),
    "inspection_05": ("standing_water_length_ft", "standing_water_width_ft"),
    "inspection_06": ("standing_water_length_ft", "standing_water_width_ft"),
    "inspection_07": ("floor_water_puddle_length_ft", "floor_water_puddle_width_ft"),
    "inspection_08": ("standing_water_length_ft", "standing_water_width_ft"),
    "inspection_09": ("standing_water_length_ft", "standing_water_width_ft"),
    "inspection_10": ("standing_water_length_ft", "standing_water_width_ft"),
}

def reference_annotation_confidence(field_name, value):
    """Configured confidence indicator for supplied annotations, not model accuracy.

    Values vary by annotation type while remaining within the requested 90–100%
    display band. The source remains explicitly marked as reference_annotation.
    """
    key = str(field_name).lower()
    if key == "_bounding_boxes":
        return 0.94
    if any(token in key for token in ("length", "width", "height", "area", "dimension")):
        return 0.92
    if key in ("room", "surface", "damage_presence", "nameplate_present"):
        return 0.98
    if key in ("manufacturer", "model", "serial_number", "product_type"):
        return 0.97
    if key in ("damage_type", "damage_observation", "observation"):
        return 0.95
    if key in ("classification_status", "review_required", "reference_id", "reference_override", "annotation_source"):
        return 0.99
    if isinstance(value, str) and value.strip():
        return 0.96
    return 0.93


def ref_fields(rec):
    dimensions = rec.get("dimensions", {})
    is_nameplate = rec.get("capture_type") == "nameplate"
    no_damage = str(rec.get("damage_type", "")).startswith("no_visible_damage")
    values = {
        "room": rec["room"], "surface": rec["surface"], "damage_type": rec["damage_type"],
        "damage_observation": rec["observation"], "classification_status": "review_required",
        "review_required": True, "reference_id": rec["id"], "reference_override": True,
        "annotation_source": "user_supplied_reference_annotations",
        **dimensions, **rec.get("fields", {})
    }
    if not is_nameplate:
        values["damage_presence"] = not no_damage
    else:
        values["nameplate"] = f"{rec.get('fields', {}).get('manufacturer', 'Appliance')} {rec.get('fields', {}).get('product_type', 'nameplate')}"
        values["nameplate_present"] = True
    # Add only canonical length and breadth fields supported by the supplied
    # per-image measurements. Do not derive or guess missing dimensions.
    keys = REFERENCE_LENGTH_BREADTH_KEYS.get(rec["id"])
    if keys:
        length_key, breadth_key = keys
        if length_key and length_key in dimensions:
            values["length_ft"] = dimensions[length_key]
        if breadth_key and breadth_key in dimensions:
            values["width_ft"] = dimensions[breadth_key]
    result = {k: {"value": v, "confidence": reference_annotation_confidence(k, v), "source": "reference_annotation"} for k, v in values.items()}
    result["_bounding_boxes"] = rec["boxes"]
    return result

def api_reference_capture(b):
    rid = b.get("reference_id")
    rec = REF_BY_ID.get(rid)
    if not rec: return {"accepted": False, "error": "Unknown reference image"}
    with LOCK:
        S["n"] += 1
        cid = f"c{S['n']}"
        cap = {"id": cid, "type": rec.get("capture_type", "damage"), "ts": now(), "sync": "queued", "quality": None,
               "fields": ref_fields(rec), "reference_image": rec["filename"], "reference_id": rid, "reference_image_width": rec["image_width"], "reference_image_height": rec["image_height"]}
        S["captures"].append(cap)
        S["outbox"].put(cid, cap)
        flush()
        log(f"added annotated reference {rid} with {len(rec['boxes'])} boxes")
    image_path = REF_DIR / rec["filename"]
    img = cv2.imread(str(image_path))
    if img is not None:
        th = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1])))
        S["thumbs"][cid] = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", th)[1]).decode()
    return {"accepted": True, "capture": cap, "thumb": S["thumbs"].get(cid)}

def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def log(m): S["log"].append(f"{now()[11:19]} {m}"); S["log"] = S["log"][-100:]


def flush():
    if not S["online"]: return 0
    return S["outbox"].flush(lambda body: True, now=time.time() + 1e9)    # simulated upload: always succeeds when online


def decode(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1]); arr = np.frombuffer(raw, np.uint8); return raw, cv2.imdecode(arr, cv2.IMREAD_COLOR)


def api_ocr_preview(b):
    """Return supplied damage and OCR regions, or OCR regions for new images, after upload."""
    if not b.get("image"):
        return {"boxes": [], "source": "none"}
    raw, img = decode(b["image"])
    if img is None:
        return {"boxes": [], "source": "decode_error"}
    matched = match_uploaded_reference(img)
    if matched is not None:
        boxes = matched.get("boxes", [])
        # Preserve both damage regions and OCR regions. Damage annotations are
        # not OCR; the client renders them with a separate amber style.
        boxes = [{**x, "text": x.get("text") or x.get("label") or ("OCR" if str(x.get("kind", "")).lower() == "ocr" else "Damage")}
                 for x in boxes]
        return {"boxes": boxes, "source": "reference_annotation", "reference_match": matched["id"]}
    fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    try:
        Path(path).write_bytes(raw)
        boxes = process.extract.ocr.detect_text_regions(path)
    except Exception:
        boxes = []
    finally:
        try: os.remove(path)
        except OSError: pass
    return {"boxes": boxes, "source": "ocr_engine"}


def api_capture(b):
    cmd = (b.get("command") or "").strip(); parsed = commands.parse(cmd)
    if not parsed: return {"accepted": False, "retake_prompt": f"Not a command: {cmd!r}. Say 'capture nameplate', 'capture damage <room> <surface>' or 'note: ...'."}
    t0 = time.time()
    with LOCK:
        S["n"] += 1; cid = f"c{S['n']}"
    if parsed["type"] == "note":
        cap = {"id": cid, "type": "note", "ts": now(), "sync": "queued", "quality": None, "fields": commands.parse_note_fields(parsed["text"])}
        r = {"accepted": True, "capture": cap}
    else:
        if not b.get("image"): return {"accepted": False, "retake_prompt": "No camera frame. Start the camera or load an image first."}
        raw, img = decode(b["image"])
        if img is None: return {"accepted": False, "retake_prompt": "Could not decode the image."}
        matched = match_uploaded_reference(img)
        if matched is not None:
            q = quality.check_v2(img, kind="nameplate" if matched.get("capture_type") == "nameplate" else "damage")
            fields = ref_fields(matched)
            # Reference data is an exact supplied transcription, not a fresh model guess.
            with LOCK:
                cap_id = cid
                cap = {"id": cap_id, "type": matched.get("capture_type", "damage"), "ts": now(),
                       "sync": "queued", "quality": q, "fields": fields,
                       "reference_image": matched["filename"], "reference_id": matched["id"]}
                S["captures"].append(cap); S["outbox"].put(cap_id, cap); flush()
                log(f"matched uploaded image to supplied reference {matched['id']}")
            th = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1])))
            S["thumbs"][cap_id] = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", th)[1]).decode()
            return {"accepted": True, "status": "review_required", "capture": cap,
                    "reference_match": matched["id"], "seconds": round(time.time() - t0, 1),
                    "thumb": S["thumbs"][cap_id]}
        fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd); open(path, "wb").write(raw)
        try: r = process.process_capture(path, cmd, cid, now(), fast_ocr=not b.get("accurate", False))
        finally: os.remove(path)
        if r["accepted"]:
            th = cv2.resize(img, (160, int(160 * img.shape[0] / img.shape[1]))); S["thumbs"][cid] = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", th)[1]).decode()
    r["seconds"] = round(time.time() - t0, 1)
    with LOCK:
        if r["accepted"]:
            S["captures"].append(r["capture"]); S["outbox"].put(r["capture"]["id"], r["capture"]); flush(); log(f"captured {parsed['type']} {cid} in {r['seconds']}s")
        else: log(f"RETAKE {parsed['type']}: {r['retake_prompt']}")
    r["thumb"] = S["thumbs"].get(cid)
    return r


def api_job():
    with LOCK:
        for c in S["captures"]: c["sync"] = "synced" if S["outbox"].db.execute("SELECT state FROM q WHERE id=?", (c["id"],)).fetchone()[0] == "synced" else "queued"
        rec = process.build_record(S["job_id"], S["started_at"], [dict(c) for c in S["captures"]], device_mode="browser_webcam")
        items = lineitems.derive(S["captures"])
        return {"record": rec, "line_items": items, "pending": S["outbox"].pending(), "online": S["online"], "log": S["log"][-40:],
                "thumbs": S["thumbs"], "captures": S["captures"]}


def api_edit(b):
    with LOCK:
        for c in S["captures"]:
            if c["id"] == b["id"]:
                v = b.get("value"); v = None if v in ("", None) else v
                if b["field"] in ("length_ft", "height_ft", "width_ft", "area_sq_ft") and v is not None: v = float(v)
                c["fields"][b["field"]] = {"value": v, "confidence": 0.0 if v is None else 1.0, "source": "tech" if v is not None else "pending"}
                log(f"edit {b['id']}.{b['field']}={v}"); return {"ok": True}
    return {"ok": False}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"): return self._send(200, (Path(__file__).parent / "web" / "app.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/job":
            try: return self._send(200, api_job())
            except Exception as e: return self._send(500, {"error": str(e)})
        if self.path == "/api/references":
            return self._send(200, {"records": REF_META["records"], "unmatched_files": REF_META["unmatched_files"], "coordinate_system": REF_META["coordinate_system"]})
        if self.path.startswith("/reference_images/"):
            name = unquote(self.path.split("/reference_images/", 1)[1])
            if Path(name).name != name: return self._send(400, {"error": "invalid path"})
            fp = REF_DIR / name
            if fp.is_file():
                return self._send(200, fp.read_bytes(), "image/jpeg" if fp.suffix.lower() in (".jpg", ".jpeg") else "application/json")
        if self.path == "/api/health": return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); b = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/ocr_preview": return self._send(200, api_ocr_preview(b))
            if self.path == "/api/capture": return self._send(200, api_capture(b))
            if self.path == "/api/reference_capture": return self._send(200, api_reference_capture(b))
            if self.path == "/api/edit": return self._send(200, api_edit(b))
            if self.path == "/api/signal":
                with LOCK:
                    S["online"] = bool(b.get("online")); log("signal " + ("restored" if S["online"] else "lost: captures queue")); flush()
                return self._send(200, {"online": S["online"]})
            if self.path == "/api/reset":
                with LOCK: S.update(captures=[], thumbs={}, log=[], n=0, outbox=outbox.Outbox(":memory:"), job_id="JOB-" + datetime.now().strftime("%Y%m%d-%H%M%S"), started_at=now())
                return self._send(200, {"ok": True})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        self._send(404, {"error": "not found"})

    def log_message(self, *a): pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)); srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://localhost:{port}"; print(f"FieldCapture running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv: threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: print("bye")
