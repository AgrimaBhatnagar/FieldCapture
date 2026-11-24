# Nameplate OCR model stack

FieldCapture uses a two engine OCR strategy. PaddleOCR provides learned text detection and recognition when installed. Tesseract remains an independent fallback and is also run alongside PaddleOCR in automatic mode. Overlapping words are aligned geometrically and textually. Agreement raises confidence; disagreements do not create a consensus value. Non overlapping detections are retained to avoid dropping small label fields.

The existing nameplate parser then applies field specific parsing, identifier plausibility checks, repeated preprocessing, and optional crop rereading. A readable field is not automatically a verified serial or model number. Ambiguous identifiers remain marked for review.

## Use an existing PaddleOCR installation

FieldCapture does not reinstall or downgrade PaddleOCR. It imports the package
already installed in the Python environment used to launch the app. The adapter
supports the common PaddleOCR 2.x and 3.x result formats and converts grayscale
preprocessing images to BGR before inference.

Check the active environment from the project root:

```bash
python tools/check_paddleocr.py
```

Run the OCR comparison against the same nameplate images:

```bash
python eval/compare_ocr_engines.py path/to/nameplate_images --output eval/ocr_engine_comparison.json
```

Set `FIELDCAPTURE_OCR_BACKEND=auto` (default) to combine PaddleOCR and
Tesseract, `paddle` to require PaddleOCR and fail visibly if it cannot run, or
`tesseract` to use only Tesseract. Make sure the app and diagnostic command use
the same Python environment where PaddleOCR is installed.

The first PaddleOCR inference may download model weights if they are not already
cached. This is separate from installing the Python package.

## Reliability boundary

No OCR accuracy percentage is claimed for PaddleOCR on this project until it is evaluated against independently transcribed nameplate images. The existing 51 image reference set includes 10 nameplate examples. Report field level exact match, character error rate, and confidence calibration on a held out set before production use.
