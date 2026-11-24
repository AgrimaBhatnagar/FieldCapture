"""Rule-based parser for voice commands and note field extraction."""
import re

ROOMS = [
    "living room", "kitchen", "bathroom", "bedroom", "basement",
    "attic", "garage", "laundry", "hallway", "closet"
]
SURFACES = [
    "wall", "ceiling", "floor", "subfloor", "cabinet",
    "baseboard", "window", "door", "drywall"
]
DIRECTIONS = ["north", "south", "east", "west", "left", "right", "front", "back"]


def parse(text: str):
    if not text or not isinstance(text, str):
        return None
    raw = text.strip()
    low = re.sub(r"\s+", " ", re.sub(r"[.,]", " ", raw.lower())).strip()
    if low.startswith("capture nameplate"):
        return {"type": "nameplate"}
    m = re.match(r"^capture damage\s*(.*)$", low)
    if m:
        toks = m.group(1).split()
        idx = max((i for i, w in enumerate(toks) if w in SURFACES), default=-1)
        if idx < 0:
            return {"type": "damage", "room": " ".join(toks) or None, "surface": None}
        start = idx - 1 if idx > 0 and toks[idx - 1] in DIRECTIONS else idx
        return {
            "type": "damage",
            "room": " ".join(toks[:start]) or None,
            "surface": " ".join(toks[start:idx + 1]),
        }
    m = re.match(r"^\s*note\s*[:,-]?\s*(.+)$", raw, re.I)
    if m:
        return {"type": "note", "text": m.group(1).strip()}
    return None


def room_confidence(room):
    return 0.0 if not room else (0.9 if room in ROOMS else 0.6)


def parse_note_fields(note_text: str) -> dict:
    low = (note_text or "").lower()
    detected_room = next((r for r in ROOMS if re.search(rf"\b{re.escape(r)}\b", low)), None)
    detected_surface = next(
        (s for s in sorted(SURFACES, key=len, reverse=True) if re.search(rf"\b{re.escape(s)}\b", low)),
        None,
    )
    condition = None
    for cond in ("soft", "wet", "leaking", "leak", "crack", "cracked", "mold", "stain", "stained", "sagging"):
        if re.search(rf"\b{re.escape(cond)}\b", low):
            condition = cond
            break
    return {
        "text": {"value": note_text, "confidence": 0.95 if note_text else 0.0, "source": "voice"},
        "room": {"value": detected_room, "confidence": room_confidence(detected_room), "source": "keyword"},
        "surface": {"value": detected_surface, "confidence": 0.85 if detected_surface else 0.0, "source": "keyword"},
        "condition": {"value": condition, "confidence": 0.8 if condition else 0.0, "source": "keyword"},
    }
