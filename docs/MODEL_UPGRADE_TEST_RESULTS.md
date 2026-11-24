# Model upgrade test results

Date: 2026-11-24

## Automated application tests

The application suite passed after the OCR verification and model workflow changes.

## Four class model readiness

The ResNet18 four class trainer and property grouped leave one property out
validation workflow are present. The current 41 image manifest contains no
verified physical property IDs and includes repeated source images in several
capture condition groups. The included sample manifest audit correctly blocks
training because it lacks the required independent properties and all four
classes. No four class checkpoint or independent property score is claimed.

## OCR comparison on the same ten nameplate images

The comparison harness was run against all ten supplied nameplate images.

* Tesseract processed 10 of 10 images, mean 0.287 seconds per image in this
  environment for the single pass used by the comparison harness.
* PaddleOCR processed 0 of 10 images because `paddleocr` is not installed.
  Installation was attempted but the environment has Python 3.13 and no
  package index network access. The dedicated Python 3.10 install path is
  documented in `requirements-vision.txt` and `docs/MODEL_UPGRADE_AND_VALIDATION.md`.
* Token agreement and accuracy are not reported because the second engine
  could not run. Existing labelled OCR results remain the prior Tesseract
  benchmark and are not relabelled as PaddleOCR results.

## Identifier verification

Model ID threshold raised to 0.78 and serial threshold to 0.82. Candidate
values below these thresholds remain unverified for technician review. This
is a confidence and syntax gate, not an external catalogue verification.
