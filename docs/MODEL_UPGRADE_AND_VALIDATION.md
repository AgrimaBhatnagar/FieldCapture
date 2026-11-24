# Damage and OCR model upgrade

## Four class damage classifier

The project includes a ResNet18 transfer learning trainer for `crack`, `stain`,
`standing_water`, and `none`. It uses leave one property out evaluation and
requires a JSONL manifest with `image`, `label`, and a real `property_id` for
every frame. A property ID must identify the physical property. It must not be
a lighting condition, folder name, room name, or capture session.

Run the data audit first:

```bash
python eval/audit_damage_dataset.py eval/damage_property_manifest.jsonl
```

Only after it passes:

```bash
python eval/train_pretrained_damage.py --manifest eval/damage_property_manifest.jsonl --evaluate-only
python eval/train_pretrained_damage.py --manifest eval/damage_property_manifest.jsonl --output models/damage_resnet18.pt
```

At least three distinct properties and all four labels are required. Each
property is held out in turn. The current bundled 41 image reference set does
not contain verified property IDs and includes repeated source photographs
under different capture conditions. Therefore it is not valid to claim
independent property validation or to train and ship a validated four class
checkpoint from that set. The training pipeline deliberately refuses to do so.

## Identifier verification

Model and serial candidates are label anchored by the OCR parser. Verification
now requires stronger confidence thresholds: 0.78 for model/product IDs and
0.82 for serial IDs, plus a plausible alphanumeric identifier shape. A
candidate below threshold remains unverified and must be reviewed. This is
not a manufacturer database check and does not prove that an ID exists.
Serial absence remains `not_visible`, not a guessed value.

## PaddleOCR and Tesseract

The PaddleOCR adapter is integrated with a graceful Tesseract fallback. Install
PaddleOCR in a dedicated Python 3.10 environment using `requirements-vision.txt`.
Then compare both engines on the same ten or more labelled nameplate images:

```bash
python eval/compare_ocr_engines.py path/to/nameplate_images --output eval/ocr_engine_comparison.json
```

The comparison records per image text, engine confidence, latency, and token
agreement. Agreement is not correctness. Use `eval/labels_nameplates.json` or
a matching ground truth file to calculate field accuracy separately.

## Environment validation status

PaddleOCR could not be installed in the build environment because it uses
Python 3.13 and package index access was unavailable. The adapter and install
instructions are included, but PaddleOCR runtime comparison is pending. The
Tesseract path is available locally. Do not claim PaddleOCR benchmark results
until the command above has run successfully.
