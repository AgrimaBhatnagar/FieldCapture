# Vision model and OCR upgrade

## Damage model

The decision tree is no longer the preferred inference path. `fieldcapture.extract.extract_damage`
loads `models/damage_resnet18.pt` (or `FIELDCAPTURE_DAMAGE_MODEL`) when present. The model is
ResNet18 with ImageNet pretrained weights and a four class inspection head. The training and
evaluation code is in `fieldcapture/pretrained_damage.py`.

Install optional dependencies:

```bash
pip install -r requirements-vision.txt
```

Create a JSONL manifest with one row per independently labelled image:

```json
{"image":"images/frame001.jpg","label":"crack","property_id":"property_001"}
```

Allowed labels: `crack`, `none`, `stain`, `standing_water`. `property_id` must be the actual
physical property identifier. Do not use image filename, archive name, room name, or source
folder as a substitute for property identity.

Run leave one property out evaluation and train the final model:

```bash
python eval/train_pretrained_damage.py --manifest eval/damage_manifest.jsonl --epochs 12
```

Evaluation is saved to `eval/damage_resnet18_property_grouped.json`. It holds out each property
in turn. Do not report a score until the manifest contains multiple properties and all four
classes are represented in the training folds. The current 41 damage labels do not include
verified property IDs, so this repository intentionally does not invent a property separated
score or train a misleading model from them.

## Uncertainty and retakes

Damage inference returns `status=uncertain` and a retake prompt when confidence is below 0.80
or the top two class probabilities are separated by less than 0.20. The existing tree fallback
has no calibrated probability margin and therefore abstains rather than pretending it is a
reliable classifier. A capture is not accepted when damage classification is uncertain.

Nameplate OCR uses PaddleOCR when installed and `FIELDCAPTURE_OCR_BACKEND=auto` (default);
Tesseract remains the fallback. Set `FIELDCAPTURE_OCR_BACKEND=paddle` to require PaddleOCR.
Model and serial fields have separate syntax and confidence checks. A field that does not pass
verification is marked uncertain and asks for a closer, sharper nameplate capture.

## What is and is not complete

Implemented in this codebase: pretrained ResNet18 training/inference path, property grouped
evaluation runner, optional PaddleOCR adapter, model and serial field checks, damage abstention
and retake response.

Not yet empirically complete: pretrained weights are downloaded by torchvision on first training
run; PaddleOCR and its runtime are optional dependencies; no property IDs or independently
verified multi property damage dataset were supplied in the current image archives. Therefore
there is no honest new model accuracy number yet. The old 65.9% tree result remains the only
measured damage baseline on the 41 labelled reference images.
