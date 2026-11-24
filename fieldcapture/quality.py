"""Frame quality metrics, retake prompts, and threshold sweeps."""
import numpy as np
from PIL import Image

DEFAULTS = {"min_sharpness": 40.0, "min_luma": 25.0, "min_framing": 0.80}


def load_grayscale(image_path, size=(160, 120)) -> np.ndarray:
    with Image.open(image_path) as im:
        return np.asarray(im.convert("L").resize(size), dtype=np.float64)


def metrics(gray: np.ndarray) -> dict:
    g = np.asarray(gray, dtype=np.float64)
    lap = (4 * g[1:-1, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:] - g[:-2, 1:-1] - g[2:, 1:-1])
    gx = np.abs(g[1:-1, 2:] - g[1:-1, :-2]) + np.abs(g[2:, 1:-1] - g[:-2, 1:-1])
    h, w = gx.shape
    centre = gx[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return {
        "sharpness": float(lap.var()),
        "luma": float(g.mean()),
        "framing": float(centre.mean() / (gx.mean() + 1e-6)),
    }


def check(gray, thresholds=None):
    t = {**DEFAULTS, **(thresholds or {})}
    m = metrics(gray)
    reasons = []
    if m["luma"] < t["min_luma"]:
        reasons.append("too dark, turn on a light")
    if m["sharpness"] < t["min_sharpness"]:
        reasons.append("blurry, hold still")
    if m["framing"] < t["min_framing"]:
        reasons.append("subject off-centre, aim at the target")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "retake_prompt": "Retake requested: " + "; ".join(reasons) if reasons else None,
        "metrics": m,
    }


def sweep(rows, key, candidates):
    out = []
    for c in candidates:
        tp = sum(1 for m, good in rows if m[key] >= c and good)
        fp = sum(1 for m, good in rows if m[key] >= c and not good)
        fn = sum(1 for m, good in rows if m[key] < c and good)
        out.append({
            "cutoff": c,
            "precision": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
        })
    return out


# ---------------------------------------------------------------------------------------------
# v2 gate (calibrated on real frames, see eval/calibrate_quality.py and eval/REPORT.md).
# v1 metrics() above are kept for the JS parity test; the web prototype still uses v1.
# ---------------------------------------------------------------------------------------------
import cv2  # noqa: E402

V2 = {"min_luma": 57.0, "max_blur_ratio": 0.70, "min_framing": 1.0}   # overwritten by calibration output


def blur_ratio(bgr_or_gray) -> float:
    """Re-blur test: gradient energy after an extra Gaussian blur / before. Sharp images lose a lot (low ratio),
    already-blurry images lose little (ratio near 1). Independent of image content and resolution."""
    g = bgr_or_gray if bgr_or_gray.ndim == 2 else cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    s = 512.0 / max(g.shape)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
    e = lambda x: float(np.abs(cv2.Sobel(x, cv2.CV_32F, 1, 0)).mean() + np.abs(cv2.Sobel(x, cv2.CV_32F, 0, 1)).mean())
    return e(cv2.GaussianBlur(g, (0, 0), 1.5)) / (e(g) + 1e-6)


def framing_ratio(gray) -> float:
    return metrics(cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA))["framing"]


def check_v2(bgr, kind="damage", t=None):
    t = {**V2, **(t or {})}
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    m = {"luma": float(g.mean()), "blur_ratio": blur_ratio(g)}
    reasons = []
    if m["luma"] < t["min_luma"]: reasons.append("too dark, turn on a light")
    if m["blur_ratio"] > t["max_blur_ratio"]: reasons.append("blurry, hold still")
    if kind == "nameplate":
        m["framing"] = framing_ratio(g)
        if m["framing"] < t["min_framing"]: reasons.append("plate off-centre, aim at the label")
    return {"pass": not reasons, "reasons": reasons,
            "retake_prompt": ("Retake requested: " + "; ".join(reasons)) if reasons else None, "metrics": m}
