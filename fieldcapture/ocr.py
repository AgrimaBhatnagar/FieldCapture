"""Offline nameplate extraction: OpenCV preprocessing + Tesseract + rule parser. No API, no network.
Confidence = (share of preprocessing variants that agree on the value) x (mean Tesseract word confidence)."""
import difflib, re
from collections import Counter, defaultdict
import cv2, numpy as np, pytesseract

BRANDS = ["LG", "GE", "WHIRLPOOL", "ELECTROLUX", "HAIER", "PANASONIC", "FUJITSU GENERAL", "FEIT ELECTRIC", "SAMSUNG", "BOSCH",
          "RHEEM", "A. O. SMITH", "BRADFORD WHITE", "STATE", "CARRIER", "TRANE", "LENNOX", "GOODMAN", "DAIKIN", "MITSUBISHI",
          "FRIGIDAIRE", "MAYTAG", "KENMORE", "NAVIEN", "RINNAI", "MIDEA", "DANBY", "HONEYWELL", "SIEMENS", "EATON", "KOHLER",
          "AMANA", "BRYANT", "YORK", "RUUD", "AMERICAN STANDARD", "TAKAGI", "BOSCH", "VAILLANT", "BAXI", "WORCESTER", "IDEAL"]
MONTHS = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
SKIP = {"NO", "NO.", "NO.:", "NO:", "N°", "#", ":", "MODELE", "MODELO", "NUMBER", "NUM", "DE", "DES", "AND", "SERIE", "SÉRIE", "SERIAL"}


def variants(path, fast=False):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    out = []
    for sc in (2.0, 3.0):
        s = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        out.append(("s%.0f" % sc, s))
        out.append(("s%.0f_clahe" % sc, cv2.createCLAHE(2.0, (8, 8)).apply(s)))
        out.append(("s%.0f_otsu" % sc, cv2.threshold(cv2.GaussianBlur(s, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))
    if fast: out = [v for v in out if v[0] in ("s2_clahe", "s3")]
    return out


def _tesseract_words(img, psm):
    d = pytesseract.image_to_data(img, config=f"--oem 1 --psm {psm}", output_type=pytesseract.Output.DICT)
    return [(t.strip(), float(c), (int(l), int(tp), int(w), int(h))) for t, c, l, tp, w, h in
            zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"])
            if t.strip() and float(c) >= 0]


def _box_iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _merge_ocr_words(paddle_rows, tess_rows):
    """Fuse overlapping PaddleOCR and Tesseract tokens without double counting.

    Paddle detections are retained when both engines see the same region. Exact
    agreement raises confidence; disagreements keep the stronger engine result
    but do not manufacture a consensus value.
    """
    out = list(paddle_rows)
    used = set()
    for out_idx, (pt, pc, pb) in enumerate(paddle_rows):
        best = None; best_score = 0.0
        for j, (tt, tc, tb) in enumerate(tess_rows):
            if j in used: continue
            overlap = _box_iou(pb, tb)
            similarity = difflib.SequenceMatcher(None, _norm(pt), _norm(tt)).ratio()
            score = overlap * 0.65 + similarity * 0.35
            if overlap >= 0.15 and score > best_score:
                best, best_score = j, score
        if best is not None:
            tt, tc, tb = tess_rows[best]; used.add(best)
            similarity = difflib.SequenceMatcher(None, _norm(pt), _norm(tt)).ratio()
            if similarity >= 0.85:
                out[out_idx] = (pt, min(100.0, max(pc, tc) + 8.0), pb)
            elif tc > pc + 12:
                out[out_idx] = (tt, tc, tb)
    # Add Tesseract detections Paddle did not localise. This preserves fallback
    # coverage for tiny text while preventing duplicate overlapping words.
    for j, row in enumerate(tess_rows):
        if j in used: continue
        if not any(_box_iou(row[2], existing[2]) >= 0.35 for existing in out):
            out.append(row)
    return out


def words(img, psm):
    """OCR ensemble: PaddleOCR plus Tesseract, with graceful Tesseract fallback."""
    backend = __import__("os").environ.get("FIELDCAPTURE_OCR_BACKEND", "auto").lower()
    tess_rows = None
    if backend in ("auto", "ensemble", "paddle"):
        try:
            from fieldcapture.ocr_backend import paddle_words
            paddle_rows = paddle_words(img)
            if backend == "paddle": return paddle_rows
            if paddle_rows:
                tess_rows = _tesseract_words(img, psm)
                return _merge_ocr_words(paddle_rows, tess_rows)
        except Exception:
            if backend == "paddle": raise
    return _tesseract_words(img, psm)


def verify_identifier_field(field, value, confidence):
    """Field specific conservative validation for model and serial identifiers."""
    if field not in ("model", "serial"):
        return {"verified": value is not None and confidence >= 0.35,
                "status": "ok" if value is not None and confidence >= 0.35 else "uncertain"}
    if field == "serial" and value is None:
        return {"verified": False, "status": "not_visible", "reason": "No serial number detected on this label"}
    cleaned = re.sub(r"[^A-Z0-9]", "", str(value or "").upper())
    min_len = 5 if field == "model" else 6
    has_digit = any(ch.isdigit() for ch in cleaned)
    alpha = any(ch.isalpha() for ch in cleaned)
    plausible = min_len <= len(cleaned) <= 24 and has_digit and alpha
    # A syntactically plausible ID is not considered verified unless OCR consensus is strong.
    # OCR confidence alone is insufficient for a serial or model identifier.
    # Require a conservative field-specific confidence and plausible structure.
    # Manufacturer catalogue lookups are intentionally not fabricated here.
    verified = bool(plausible and confidence >= (0.78 if field == "model" else 0.82))
    return {"verified": verified, "status": "verified" if verified else "uncertain",
            "reason": None if verified else "Identifier is a candidate only. Confirm the printed label, improve focus and lighting, and compare all characters before accepting."}




def _clean(t):
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9)]+$", "", t)


def _valid_id(t, need_digit=True):
    t = _clean(t)
    if len(t) < 5 or len(t) > 24 or t.upper() in SKIP or "." in t or "WWW" in t.upper() or t.upper().endswith("COM"):
        return False
    if need_digit and not any(c.isdigit() for c in t):
        return False
    return sum(c.isalnum() or c in "-/()" for c in t) / len(t) > 0.85


LABELWORDS = {"MODEL", "MODELE", "MODELO", "NO", "SERIAL", "SER", "ITEM", "MOD", "SN", "S/N"}


def _is_label(tok, labels):
    u = re.sub(r"[^A-Z/]", "", tok.upper())
    return any(u == l or (l in ("MODEL", "SERIAL") and u.startswith(l)) for l in labels)


def _after(ws, labels, window=4):
    for i, w_ in enumerate(ws):
        t = w_[0]
        if not _is_label(t, labels): continue
        for j in range(i + 1 if re.sub(r"[^A-Za-z/]", "", t) else i, min(len(ws), i + window + 1)):
            for sub in ws[j][0].split(":"):
                c = _clean(sub)
                if not c or _is_label(c, labels | LABELWORDS) or not _valid_id(c): 
                    if c and _is_label(c, labels | LABELWORDS): continue
                    # allow short alpha prefix + id (e.g. "WRF 767SDEN@1")
                    if c.isalpha() and 2 <= len(c) <= 4 and c.upper() not in LABELWORDS and j + 1 < len(ws) and _valid_id(_clean(ws[j + 1][0])):
                        return c.upper() + _clean(ws[j + 1][0]), ws[j + 1][1], (j, j + 1)
                    continue
                conf = ws[j][1]; span = (j, j)
                if j + 1 < len(ws):  # join split model numbers like "AU082F YCRA(HW)"
                    nx = _clean(ws[j + 1][0])
                    if nx and ("(" in ws[j + 1][0] or any(ch.isdigit() for ch in nx)) and len(nx) <= 10 and nx.upper() not in SKIP and not _is_label(nx, LABELWORDS) and not nx.endswith(":") and len(c) >= 5 and any(ch.isalpha() for ch in c) and "(" in ws[j + 1][0]:
                        c = c + ws[j + 1][0].strip(".,;:"); span = (j, j + 1)
                return (c.upper() if any(ch.isdigit() for ch in c) else c), conf, span
    return None, 0.0, None


LABELFRAG = ("MANU", "MFG", "MFR", "DATE", "SERIAL", "MODEL", "SER", "MOD", "NO")


def strip_label_prefix(v):
    """Remove a merged label fragment from the front of a value (e.g. 'MANUHG200312N' -> 'HG200312N')."""
    u = v.upper()
    for l in sorted(LABELFRAG, key=len, reverse=True):
        rest = u[len(l):]
        if u.startswith(l) and len(v) >= 8 and len(rest) >= 5 and any(c.isdigit() for c in rest):
            return v[len(l):]
    return v


def parse_words(ws):
    f = {}
    text = " ".join(w_[0] for w_ in ws).upper()
    best = (0, None)
    toks = [re.sub(r"[^A-Z. ]", "", w_[0].upper()) for w_ in ws]
    cands = set(toks) | {" ".join(toks[i:i + 2]) for i in range(len(toks) - 1)}
    for b in BRANDS:
        for c in cands:
            if len(c) < 2: continue
            r = difflib.SequenceMatcher(None, b, c.strip()).ratio()
            if (len(b) <= 3 and c.strip() == b) or (len(b) > 3 and r > best[0] and r >= 0.8):
                if r > best[0]: best = (r, b)
    if best[1]: f["make"] = (best[1], 1.0 * 100)
    m, c, sp = _after(ws, {"MODEL", "MOD"})
    if not m: m, c, sp = _after(ws, {"ITEM"})
    if m: f["model"] = (strip_label_prefix(m), c); f["_span_model"] = sp
    s, c, sp = _after(ws, {"SERIAL", "SER", "S/N", "SN"}, window=5)
    if s: f["serial"] = (strip_label_prefix(s), c); f["_span_serial"] = sp
    dm = re.search(rf"\b((?:{MONTHS})[A-Z]*)[\s.,]*((?:19|20)\d\d)\b", text)
    if dm: f["manufacture_date"] = (f"{dm.group(1)} {dm.group(2)}", 60.0)
    else:
        dm = re.search(r"\b((?:19|20)\d\d)[./-](0?[1-9]|1[0-2])\b", text) or re.search(r"DATE[^0-9]{0,6}((?:0?[1-9]|1[0-2])[/-]\d{2,4})\b", text)
        if dm: f["manufacture_date"] = (dm.group(1) if dm.lastindex == 1 else f"{dm.group(1)}.{dm.group(2)}", 50.0)
    return f


def _norm(v): return re.sub(r"[^A-Z0-9]", "", str(v).upper())


WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/()"


def _union(ws, span, scale):
    bs = [ws[j][2] for j in range(span[0], span[1] + 1) if len(ws[j]) > 2]
    if not bs: return None
    x0 = min(b[0] for b in bs); y0 = min(b[1] for b in bs); x1 = max(b[0] + b[2] for b in bs); y1 = max(b[1] + b[3] for b in bs)
    return (x0 / scale, y0 / scale, (x1 - x0) / scale, (y1 - y0) / scale)


def refine_value(gray, box, original):
    """Re-read ONE field's pixels at high zoom with a restricted character set (no '@', no lowercase), several
    preprocessings, majority vote. Accepts only if >=3 of 6 reads agree and are similar to the first-pass value."""
    x, y, w, h = box; pad = int(0.4 * h) + 2
    crop = gray[max(0, int(y - pad)):int(y + h + pad), max(0, int(x - 2 * pad)):int(x + w + 2 * pad)]
    if crop.size == 0: return None, 0.0
    reads = []
    for sc in (4, 6):
        r = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        for v in (r, cv2.createCLAHE(2.0, (4, 4)).apply(r), cv2.threshold(cv2.GaussianBlur(r, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]):
            v = cv2.copyMakeBorder(v, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
            t = pytesseract.image_to_string(v, config=f"--oem 1 --psm 7 -c tessedit_char_whitelist={WL}").strip().replace(" ", "")
            if _valid_id(t) or (len(t) >= 5 and any(ch.isdigit() for ch in t)): reads.append(t)
    if not reads: return None, 0.0
    cnt = Counter(_norm(r) for r in reads); key, n = cnt.most_common(1)[0]
    sim = difflib.SequenceMatcher(None, key, _norm(original)).ratio()
    if n >= 3 and sim >= 0.6 and abs(len(key) - len(_norm(original))) <= 3:
        return next(r for r in reads if _norm(r) == key), n / 6.0
    return None, 0.0


def detect_text_regions(path, max_regions=80):
    """Return normalized OCR word boxes for visible text in an uploaded image.

    Coordinates use the project's normalized [ymin, xmin, ymax, xmax] format.
    OCR scores are not promoted to model confidence; the boxes simply localize text.
    """
    image = cv2.imread(str(path))
    if image is None:
        return []
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows = []
    try:
        rows = _tesseract_words(gray, 11)
    except Exception:
        rows = []
    # Prefer readable, nontrivial text and avoid drawing tiny noise boxes.
    out = []
    for text, score, (x, y, w, h) in rows:
        text = str(text).strip()
        alnum = sum(ch.isalnum() for ch in text)
        if not text or len(text) < 3 or w < 7 or h < 6 or score < 55 or alnum / max(1, len(text)) < 0.70:
            continue
        x1 = max(0, min(998, round(x / max(1, width) * 998)))
        y1 = max(0, min(998, round(y / max(1, height) * 998)))
        x2 = max(x1 + 1, min(998, round((x + w) / max(1, width) * 998)))
        y2 = max(y1 + 1, min(998, round((y + h) / max(1, height) * 998)))
        out.append({"kind": "ocr", "text": text, "box": [y1, x1, y2, x2]})
        if len(out) >= max_regions:
            break
    return out


def extract_nameplate(path, fast=False, refine=False):
    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    votes = defaultdict(lambda: defaultdict(list)); n = 0
    for name, img in variants(path, fast):
        sc = 3.0 if name.startswith("s3") else 2.0
        for psm in (11, 6):
            n += 1
            try:
                ws = words(img, psm); f = parse_words(ws)
            except Exception: continue
            for k, (v, c) in [(k, v) for k, v in f.items() if not k.startswith("_")]:
                box = _union(ws, f["_span_" + k], sc) if ("_span_" + k) in f else None
                votes[k][_norm(v)].append((v, c, box))
    out = {}
    for k, d in votes.items():
        key, lst = max(d.items(), key=lambda kv: len(kv[1]))
        share = len(lst) / n
        mean_word_conf = sum(c for _, c, _ in lst) / len(lst) / 100.0
        conf = share * mean_word_conf
        # For identifier fields, several agreeing OCR passes with strong token-level
        # confidence are more informative than the raw fraction of all preprocessing
        # variants. The latter unfairly penalises labels where thresholding variants
        # fail while the original and enlarged views agree.
        if k in ("model", "serial") and len(lst) >= 3:
            conf = max(conf, 0.75 * mean_word_conf)
        value = Counter(v for v, _, _ in lst).most_common(1)[0][0]; src = "ocr"
        boxes = [b for _, _, b in lst if b]
        if refine and k in ("model", "serial") and boxes:
            rv, rshare = refine_value(gray, max(boxes, key=lambda b: b[2] * b[3]), value)
            if rv:
                src = "ocr+crop"
                if _norm(rv) != key: value = rv; conf = 0.5 * rshare   # disagreement with first pass: lower trust
                else: conf = min(1.0, conf + 0.25 * rshare)
        out[k] = {"value": value, "confidence": round(min(conf, 1.0), 3), "source": src}
    for k in ("make", "model", "serial", "manufacture_date"):
        out.setdefault(k, {"value": None, "confidence": 0.0, "source": "ocr"})
    # A serial number is not present on every product label. Missing serials are
    # recorded as absent, not treated as OCR failure. ITEM / PRODUCT identifiers
    # are valid product identifiers even when the label does not say MODEL.
    if out["model"]["value"] is None:
        # Some appliance and product cartons use ITEM #, PRODUCT #, CATALOG NO. or SKU.
        # parse_words already recognises ITEM; this status makes absence explicit.
        out["model"]["verification"] = {"verified": False, "status": "not_found"}
    else:
        out["model"]["verification"] = verify_identifier_field("model", out["model"]["value"], out["model"]["confidence"])
    if out["serial"]["value"] is None:
        out["serial"]["verification"] = {"verified": False, "status": "not_visible", "reason": "No serial number detected on this label"}
    else:
        out["serial"]["verification"] = verify_identifier_field("serial", out["serial"]["value"], out["serial"]["confidence"])
    # A readable manufacturer plus a verified model/product identifier is enough
    # to save a partial record. Serial is optional unless explicitly required later.
    out["requires_review"] = bool(out["model"]["value"] and not out["model"]["verification"]["verified"])
    out["retake_prompt"] = ("Product identifier could not be verified. Move closer, reduce glare, hold steady, and capture the full label."
                            if out["requires_review"] else None)
    out["_bounding_boxes"] = detect_text_regions(path)
    return out
