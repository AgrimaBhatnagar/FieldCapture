"""End-to-end walkthrough job builder."""
from fieldcapture import commands, lineitems, quality, validate


def build_job_record(job_id: str, started_at: str, session_ctrl, walkthrough_items: list[dict], thresholds=None) -> dict:
    captures = []
    for idx, item in enumerate(walkthrough_items, 1):
        cid = item.get("id", f"cap-{idx:03d}")
        ctype = item["type"]
        ts = item.get("ts", started_at)
        sync = item.get("sync", "queued")
        q_res = None
        if item.get("gray") is not None:
            q_res = quality.check(item["gray"], thresholds=thresholds)
        if ctype == "note":
            fields = commands.parse_note_fields(item.get("note_text", ""))
        else:
            fields = dict(item.get("fields") or {})
        captures.append({
            "id": cid,
            "type": ctype,
            "ts": ts,
            "sync": sync,
            "quality": q_res,
            "fields": fields,
        })
    items_out = lineitems.derive(captures)
    job = {
        "schema": "fieldcapture.job/0.1",
        "job_id": job_id,
        "started_at": started_at,
        "session": {
            "state": session_ctrl.state,
            "device_mode": session_ctrl.device_mode,
            "drop_recoveries": session_ctrl.drop_recoveries,
        },
        "captures": captures,
        "line_items": [
            {"code": li["code"], "qty": li["qty"], "confidence": li["confidence"]}
            for li in items_out
        ],
    }
    errs = validate.validate(job)
    if errs:
        raise ValueError(f"Invalid job record: {errs}")
    return job
