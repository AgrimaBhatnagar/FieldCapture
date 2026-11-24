#!/usr/bin/env python3
"""Rebuild the trained binary damage presence Random Forest.

The bundled 41 images are development references, not an independent test set.
The report uses leave one out cross validation and must not be presented as a
held out or real world performance result.
"""
import json
import pathlib
import pickle
import sys

import cv2
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from fieldcapture.damage import features


def main():
    ref_dir = ROOT / "reference_damage"
    manifest = json.loads((ref_dir / "manifest.json").read_text(encoding="utf-8"))
    X, y = [], []
    for row in manifest:
        image = cv2.imread(str(ref_dir / row["file"]))
        if image is None:
            raise RuntimeError(f"Cannot read {row['file']}")
        X.append(features(image))
        y.append("none" if row["label"] == "none" else "damage")
    X, y = np.asarray(X), np.asarray(y)
    model = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=42,
    )
    oof_prob = []
    for train, test in LeaveOneOut().split(X):
        fitted = clone(model).fit(X[train], y[train])
        oof_prob.append(fitted.predict_proba(X[test])[0])
    oof_prob = np.asarray(oof_prob)
    classes = fitted.classes_
    pred = classes[oof_prob.argmax(axis=1)]
    conf = oof_prob.max(axis=1)
    threshold = 0.70
    keep = conf >= threshold
    selective_accuracy = float(accuracy_score(y[keep], pred[keep])) if keep.any() else None
    report = {
        "task": "binary damage presence detection",
        "dataset_size": int(len(y)),
        "labels_source": "bundled development labels, not independently verified",
        "validation": "leave one out cross validation",
        "overall_accuracy": round(float(accuracy_score(y, pred)), 4),
        "selective_threshold": threshold,
        "selective_coverage_count": int(keep.sum()),
        "selective_coverage_fraction": round(float(keep.mean()), 4),
        "selective_accuracy": round(selective_accuracy, 4) if selective_accuracy is not None else None,
        "abstained_count": int((~keep).sum()),
        "confusion_all": confusion_matrix(y, pred, labels=["damage", "none"]).tolist(),
        "warning": "Development cross validation only. Not an independent held out score or evidence of field performance.",
    }
    model.fit(X, y)
    out = PROJECT / "models" / "damage_presence_rf.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(model, f)
    (ROOT / "damage_presence_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(out), **report}, indent=2))


if __name__ == "__main__":
    main()
