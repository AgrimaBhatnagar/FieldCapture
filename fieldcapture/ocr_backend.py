"""Optional PaddleOCR adapter compatible with PaddleOCR 2.x and 3.x.

The user's existing PaddleOCR installation is used as-is. No package is
reinstalled or downgraded by the application. Tesseract remains the fallback.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _engine():
    from paddleocr import PaddleOCR

    # PaddleOCR 2.x accepts use_angle_cls/show_log. PaddleOCR 3.x removed
    # those options, so retry with its simpler constructor when necessary.
    try:
        return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except Exception as first_error:
        try:
            return PaddleOCR(lang="en")
        except Exception as second_error:
            raise RuntimeError(
                "PaddleOCR is installed but could not initialise. "
                f"2.x style error: {first_error}; fallback constructor error: {second_error}"
            ) from second_error


def _as_bgr(image):
    import cv2
    import numpy as np

    if isinstance(image, str):
        arr = cv2.imread(image)
    else:
        arr = np.asarray(image) if image is not None else None
    if arr is None:
        return None
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape for PaddleOCR: {arr.shape}")
    return arr


def _parse_v2(raw):
    out = []
    if not isinstance(raw, list):
        return out
    # PaddleOCR 2.x normally returns [ [ [box, (text, score)], ... ] ].
    rows = raw
    if len(raw) == 1 and isinstance(raw[0], list):
        rows = raw[0]
    for item in rows or []:
        try:
            box, rec = item[0], item[1]
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            out.append((str(rec[0]).strip(), float(rec[1]) * 100.0,
                        (int(min(xs)), int(min(ys)), int(max(xs)-min(xs)), int(max(ys)-min(ys)))))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return out


def _parse_v3(raw):
    import numpy as np

    out = []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    for item in raw:
        data = item if isinstance(item, dict) else getattr(item, "json", {})
        if callable(data):
            data = data()
        if not isinstance(data, dict):
            continue
        data = data.get("res", data)
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        polys = data.get("dt_polys", data.get("rec_polys", []))
        for text, score, poly in zip(texts, scores, polys):
            try:
                pts = np.asarray(poly, dtype=float).reshape(-1, 2)
                xs, ys = pts[:, 0], pts[:, 1]
                out.append((str(text).strip(), float(score) * 100.0,
                            (int(xs.min()), int(ys.min()), int(xs.max()-xs.min()), int(ys.max()-ys.min()))))
            except (TypeError, ValueError, IndexError):
                continue
    return out


def paddle_words(image):
    """Return (text, confidence_percent, xywh) records for the OCR ensemble."""
    arr = _as_bgr(image)
    if arr is None:
        return []
    engine = _engine()
    if hasattr(engine, "ocr"):
        try:
            raw = engine.ocr(arr, cls=True)
        except TypeError:
            raw = engine.ocr(arr)
        rows = _parse_v2(raw)
        if rows:
            return [row for row in rows if row[0]]
    if hasattr(engine, "predict"):
        raw = engine.predict(arr)
        rows = _parse_v3(raw)
        if rows:
            return [row for row in rows if row[0]]
    return []
