"""Schema validator for fieldcapture.job/0.1 records."""
import json, pathlib, re

SCHEMA = json.loads((pathlib.Path(__file__).parent.parent / "schema.json").read_text(encoding="utf-8"))
CODE_PATTERN = re.compile(SCHEMA["properties"]["line_items"]["items"]["properties"]["code"]["pattern"])


def validate(rec):
    """Validate the job envelope while allowing UI-only capture metadata.

    Reference captures keep ``_bounding_boxes`` inside ``fields`` for the
    browser overlay. That value is a list, not a schema field object, and must
    not be treated as a confidence-bearing field.
    """
    errs = []
    if not isinstance(rec, dict):
        return ["job record must be an object"]
    for k in SCHEMA["required"]:
        if k not in rec:
            errs.append(f"missing {k}")
    if rec.get("schema") != "fieldcapture.job/0.1":
        errs.append("bad schema id")
    sess = rec.get("session") or {}
    if not isinstance(sess, dict):
        errs.append("session must be an object")
        sess = {}
    if sess.get("state") not in ("idle", "active", "paused", "stopped"):
        errs.append("bad session.state")
    if sess.get("device_mode") not in ("mock_device_kit", "phone_camera_fallback", "browser_webcam"):
        errs.append("bad session.device_mode")

    captures = rec.get("captures", [])
    if not isinstance(captures, list):
        errs.append("captures must be an array")
        captures = []
    for c in captures:
        if not isinstance(c, dict):
            errs.append("capture entry must be an object")
            continue
        if c.get("type") not in ("nameplate", "damage", "note"):
            errs.append(f"{c.get('id')}: bad type")
        if c.get("sync") not in ("queued", "synced"):
            errs.append(f"{c.get('id')}: bad sync")
        fields = c.get("fields") or {}
        if not isinstance(fields, dict):
            errs.append(f"{c.get('id')}.fields must be an object")
            continue
        for k, f in fields.items():
            # UI metadata such as normalized OCR/damage bounding boxes is
            # deliberately stored alongside fields for preview rendering.
            if k.startswith("_"):
                continue
            if not isinstance(f, dict):
                errs.append(f"{c.get('id')}.{k}: field must be an object")
                continue
            confidence = f.get("confidence", -1)
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                errs.append(f"{c.get('id')}.{k}: confidence out of range")

    items = rec.get("line_items", [])
    if not isinstance(items, list):
        errs.append("line_items must be an array")
        items = []
    for item in items:
        if not isinstance(item, dict):
            errs.append("line item must be an object")
            continue
        if not CODE_PATTERN.match(item.get("code", "")):
            errs.append(f"bad code {item.get('code')}")
        confidence = item.get("confidence", -1)
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errs.append(f"{item.get('code')}: confidence out of range")
    return errs
