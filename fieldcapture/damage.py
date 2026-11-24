"""Damage-type classifier from classical CV features + a small decision tree. No API, no deep learning.
Classes: crack | stain (water stain, peeling, mould) | standing_water | none."""
import cv2, numpy as np

FEATURES = ["crack_len", "crack_frac", "brown", "brown_low", "smooth_brown", "green", "dark", "bright_low", "edge_density"]


def features(bgr):
    s = 384.0 / max(bgr.shape[:2]); im = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); h, w = g.shape
    bh = cv2.morphologyEx(cv2.GaussianBlur(g, (3, 3), 0), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = (bh > 22).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    L = 0.0; px = 0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]; l = max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        if l >= 0.12 * max(h, w) and a / l <= 6: L += l; px += a
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    brown = ((H >= 5) & (H <= 25) & (S >= 60) & (V >= 40) & (V <= 230))
    m = cv2.blur(g.astype(np.float32), (7, 7)); m2 = cv2.blur((g.astype(np.float32)) ** 2, (7, 7)); sd = np.sqrt(np.maximum(m2 - m * m, 0))
    low = slice(h // 2, h)
    return [L / max(h, w), px / (h * w), brown.mean(), brown[low].mean(), (brown & (sd < 6)).mean(),
            ((H >= 28) & (H <= 85) & (S >= 45) & (V >= 40)).mean(), (V < 40).mean(), (V[low] > 235).mean(),
            (cv2.Canny(g, 60, 150) > 0).mean()]


def region_bbox(bgr, damage_class):
    """Rough pixel bounding box (x, y, w, h) of the damaged region in ORIGINAL image pixels, or None.
    stain/standing_water: largest brown blob. crack: union of long thin dark ridges. Heuristic, unvalidated."""
    H0, W0 = bgr.shape[:2]; s = 384.0 / max(H0, W0); im = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if damage_class == "crack":
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        bh = cv2.morphologyEx(cv2.GaussianBlur(g, (3, 3), 0), cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n, _, st, _ = cv2.connectedComponentsWithStats((bh > 22).astype(np.uint8), connectivity=8)
        boxes = [st[i] for i in range(1, n) if max(st[i, 2], st[i, 3]) >= 0.12 * max(g.shape) and st[i, 4] / max(st[i, 2], st[i, 3]) <= 6]
        if not boxes: return None
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes); x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    else:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        m = (((H >= 5) & (H <= 25) & (S >= 60) & (V >= 40) & (V <= 230))).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: return None
        x0, y0, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea)); x1, y1 = x0 + w, y0 + h
        if (x1 - x0) * (y1 - y0) < 0.01 * im.shape[0] * im.shape[1]: return None
    return (int(x0 / s), int(y0 / s), int((x1 - x0) / s), int((y1 - y0) / s))
