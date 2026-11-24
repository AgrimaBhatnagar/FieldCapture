# FieldCapture vision upgrade, implementation and verification

## Code changes in this build

1. Damage classifier: added an ImageNet pretrained ResNet18 transfer learning implementation in `fieldcapture/pretrained_damage.py`. It supports model training, inference, and leave one property out evaluation. `fieldcapture.extract.extract_damage` prefers a trained checkpoint at `models/damage_resnet18.pt` or the path in `FIELDCAPTURE_DAMAGE_MODEL`.
2. Property separated evaluation: added `eval/train_pretrained_damage.py`. The runner requires a JSONL manifest with actual `property_id` values and evaluates with LeaveOneGroupOut. It refuses to run with fewer than three groups.
3. PaddleOCR: added an optional PaddleOCR adapter in `fieldcapture/ocr_backend.py`. It is used when PaddleOCR is installed, with Tesseract as fallback. Set `FIELDCAPTURE_OCR_BACKEND=paddle` to require PaddleOCR.
4. Identifier verification: model and serial number fields now have separate syntax and confidence thresholds. Unverified fields are marked uncertain and nameplate capture returns a retake prompt rather than silently accepting them.
5. Damage uncertainty: damage predictions below confidence 0.80 or top class margin 0.20 abstain and request a new image. The legacy decision tree is only a conservative fallback and does not claim calibrated confidence.

## Verification performed in this environment

- Python compilation: all Python files compiled successfully.
- Unit tests: 56 passed.
- Existing 51 reference images: the earlier fresh run measured 24/30 exact nameplate fields (80%) and 27/41 damage predictions correct (65.9%) using the legacy decision tree.
- The 51 image quality gate accepted 46/51 frames, including 5/10 low light frames.
- Individual image results and the detailed reference test JSON are included under `eval/reference_test_51/`.

## Important limitations, not hidden

The 41 damage labels in the supplied image collections do not contain verified physical property identifiers. The folders identify source collections and lighting/content groups, not actual properties. Using those folder names as property IDs would leak the split and would not be a valid property separated evaluation.

The runtime used for this build does not have PaddleOCR or PaddlePaddle installed, and no ImageNet ResNet18 weights are cached. The new integration code and training pipeline are included, but the pretrained model has not been trained or measured here. On a machine with internet access, install `requirements-vision.txt`; torchvision downloads the ImageNet weights on first training run. Install PaddleOCR using the same file to activate the Paddle backend.

No new pretrained damage accuracy is claimed. The 65.9% figure is the old decision tree baseline on the small reference set, not the new ResNet18 model and not production accuracy.

## Training and evaluation

Prepare `eval/damage_manifest.jsonl` using the schema in `eval/damage_manifest.template.jsonl`. Each row needs:
- `image`: path to the image
- `label`: one of `crack`, `none`, `stain`, `standing_water`
- `property_id`: actual physical property identifier

Then run:

```bash
pip install -r requirements-vision.txt
python eval/train_pretrained_damage.py --manifest eval/damage_manifest.jsonl --epochs 12
```

The property separated report is written to `eval/damage_resnet18_property_grouped.json`. The final model is saved to `models/damage_resnet18.pt`.

Do not use images from the same property in both training and test folds. Do not use archive folder names as property IDs. Keep an independent external test set for the final submission.
