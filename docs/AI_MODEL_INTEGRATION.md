# AI model integration update

## Damage prediction

The shipped damage detector remains a trained Random Forest binary classifier, trained on 41 development images. Runtime decisions now use conservative asymmetric thresholds: 0.85 confidence for a damage flag, 0.90 for a no damage decision, and a 0.20 probability margin. Borderline predictions abstain for technician review. This is a safety policy change, not evidence of improved model accuracy. The current data is too small and lacks independently verified property separated labels to support a reliable production claim or a four class deep learning model.

## Nameplate OCR

Automatic OCR now fuses PaddleOCR learned text recognition with Tesseract. Overlapping detections are matched by geometry and text similarity. Agreement raises confidence, disagreement is not treated as consensus, and unmatched text detections are retained. If PaddleOCR is unavailable, Tesseract continues to work as a fallback.

Install optional OCR dependencies with `pip install -r requirements-vision.txt`. PaddleOCR weights may download at first use. PaddleOCR was not installed in the build environment, so the dual engine path is implemented but not runtime validated here. Tesseract fallback is covered by the automated tests.

## Verification boundary

The project test suite and Python compilation pass. No new accuracy score is claimed for either model. A production quality claim requires a larger independently labelled dataset, property separated validation, exact nameplate transcription, and field testing on the target camera hardware.
