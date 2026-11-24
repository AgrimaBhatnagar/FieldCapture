"""12-code estimate catalog, rule mapping, and depth-free size estimation."""
import math

CATALOG = {
    "DRY-01": ("Remove and replace drywall", "sq ft"),
    "DRY-02": ("Prime and paint", "sq ft"),
    "CEI-01": ("Remove and replace ceiling drywall", "sq ft"),
    "FLR-01": ("Replace subfloor section", "sq ft"),
    "MLD-01": ("Mold remediation", "sq ft"),
    "INS-01": ("Remove wet insulation", "sq ft"),
    "BSB-01": ("Replace baseboard", "lin ft"),
    "CAB-01": ("Detach and reset cabinet", "each"),
    "DOC-01": ("Moisture mapping", "per room"),
    "DEH-01": ("Dehumidifier, days to confirm", "per room"),
    "AIR-01": ("Air mover, days to confirm", "per room"),
    "WH-01": ("Water heater inspection", "each"),
}
WET = {"water stain", "sagging", "soft or rot"}


def _val(c, k):
    return (c.get("fields", {}).get(k) or {}).get("value")


def _conf(c, k):
    return (c.get("fields", {}).get(k) or {}).get("confidence", 0.0)


def area(c):
    l, h = _val(c, "length_ft"), _val(c, "height_ft")
    return round(l * h, 1) if l and h else None


def estimate_size_without_depth(
    frame_width_px: int,
    frame_height_px: int,
    bbox_width_px: float,
    bbox_height_px: float,
    assumed_dist_ft: float = 4.0,
    hfov_deg: float = 65.0,
):
    if not frame_width_px or not frame_height_px:
        return None, None, 0.0
    scene_w_ft = 2.0 * assumed_dist_ft * math.tan(math.radians(hfov_deg / 2.0))
    scene_h_ft = scene_w_ft * (frame_height_px / frame_width_px)
    est_w = round((bbox_width_px / frame_width_px) * scene_w_ft, 2)
    est_h = round((bbox_height_px / frame_height_px) * scene_h_ft, 2)
    return est_w, est_h, 0.45


def derive(captures):
    out, rooms = [], set()

    def add(code, qty, basis, conf):
        d, u = CATALOG[code]
        out.append({
            "code": code,
            "description": d,
            "unit": u,
            "qty": qty,
            "basis": basis,
            "confidence": round(conf, 2),
        })

    for c in captures:
        t = c["type"]
        if t == "nameplate":
            add("WH-01", 1, f"{c['id']}: nameplate captured; confirm appliance", 0.5)
        elif t == "damage" and _val(c, "damage_type"):
            dt, surf = _val(c, "damage_type"), (_val(c, "surface") or "")
            s, a = surf.split(" ")[-1], area(c)
            cf = min(_conf(c, "surface") or 0.5, 1.0 if a else 0.3)
            b = f"{c['id']}: {dt} on {surf or '?'}" + ("" if a else " (size needed)")
            if s == "ceiling":
                add("CEI-01", a, b, cf)
            elif s in ("subfloor", "floor"):
                add("FLR-01", a, b, cf)
            elif s == "baseboard":
                add("BSB-01", _val(c, "length_ft"), b, cf)
            elif s == "cabinet":
                add("CAB-01", 1, b, cf)
            else:
                add("DRY-01", a, b, cf)
            if s in ("wall", "ceiling", "drywall"):
                add("DRY-02", a, b, cf)
            if dt == "mold":
                add("MLD-01", a, b, cf)
            if dt == "sagging" and s in ("wall", "ceiling"):
                add("INS-01", a, b, cf)
            room = _val(c, "room") or "unknown room"
            if room not in rooms:
                rooms.add(room)
                add("DOC-01", 1, f"room: {room}", 0.8)
                if dt in WET:
                    add("DEH-01", 1, f"room: {room}", 0.6)
                    add("AIR-01", 1, f"room: {room}", 0.6)
        elif t == "note":
            txt = (_val(c, "text") or "").lower()
            if "cabinet" in txt:
                add("CAB-01", 1, f"{c['id']}: note mentions cabinet", 0.5)
            if "subfloor" in txt and "soft" in txt:
                add("FLR-01", None, f"{c['id']}: soft subfloor, size needed", 0.4)
    return out
