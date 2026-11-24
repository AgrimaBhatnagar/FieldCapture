#!/usr/bin/env python3
"""Check the PaddleOCR installation in the active Python environment."""
import sys
try:
    import paddle
    import paddleocr
    from paddleocr import PaddleOCR
    print(f"Python: {sys.version.split()[0]}")
    print(f"PaddlePaddle: {getattr(paddle, '__version__', 'unknown')}")
    print(f"PaddleOCR: {getattr(paddleocr, '__version__', 'unknown')}")
    print("Imports: OK")
    print("Next: python eval/compare_ocr_engines.py <nameplate_image_folder>")
except Exception as exc:
    print(f"PaddleOCR check failed: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
