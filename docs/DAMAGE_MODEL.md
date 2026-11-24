# Trained damage model

FieldCapture now loads `models/damage_presence_rf.pkl` during damage capture.
It is a trained scikit learn Random Forest that predicts the binary classes
`damage` and `none` from nine OpenCV image features. The checkpoint was trained
on 41 labelled development images.

## Runtime behaviour

1. The image quality gate runs before classification.
2. The Random Forest estimates damage presence.
3. Damage predictions below 0.85 confidence or below 0.20 probability margin abstain. No damage predictions require 0.90 confidence and the same margin. This asymmetric policy is deliberately conservative
   and request a clearer capture.
4. A sufficiently confident `none` prediction is recorded as no detected damage. Borderline negative predictions abstain rather than silently clearing a surface.
5. A confident `damage` prediction receives an advisory subtype from the older
   four class tree. The subtype is explicitly marked for technician review and
   must not be treated as a confirmed estimate category.

## Reported development results

The bundled leave one out cross validation report records 78.05% overall
accuracy on 41 development images. At a 0.70 confidence threshold, the
historical run accepted 24 of 41 predictions and reported 95.83% accuracy on
that accepted subset. This selective score is not overall accuracy, and the
accepted subset was biased toward damage positive predictions. Labels were not
independently verified and the same small reference collection was used to
train the shipped checkpoint. These figures are not a held out test score or
proof of real world performance.

## Rebuild

From the project root:

```bash
python eval/train_damage_presence.py
```

The script reads `eval/reference_damage/manifest.json`, rebuilds the model,
and writes `models/damage_presence_rf.pkl` plus the development report.

## Known limitations

The trained model detects damage presence only. Crack, stain, and standing
water are not independently learned by this checkpoint. The subtype suggestion
is from the legacy feature tree and requires human confirmation. It does not
measure damage dimensions. Collect independently reviewed images across
multiple properties and evaluate with property separated splits before making
production accuracy claims.


## Conservative decision policy update

The runtime now uses separate thresholds for the two binary outcomes: 0.85 for a damage flag and 0.90 for a no damage decision, plus a 0.20 probability margin. These are safety oriented operating thresholds, not a claim that the model has achieved those accuracy levels. The small development set does not support a reliable production accuracy estimate. Uncertain cases are routed to technician review.
