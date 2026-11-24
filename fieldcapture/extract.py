"""Offline extraction entry point. NO API, NO network: OpenCV + Tesseract + scikit-learn only."""
import json, pathlib, pickle
import cv2
from fieldcapture import damage, ocr

_RESULTS = pathlib.Path(__file__).parent.parent / "eval" / "damage_results.json"
_TREE = pathlib.Path(__file__).parent.parent / "eval" / "damage_tree.pkl"
TYPE_MAP = {"stain": "water stain", "crack": "crack", "standing_water": "water stain", "none": None}


def extract_nameplate(path, fast=False):
    return ocr.extract_nameplate(path, fast=fast)


def extract_damage(bgr, confidence_threshold=0.85, margin_threshold=0.20, no_damage_threshold=0.90):
    """Run the trained Random Forest damage presence model, then suggest a subtype.

    The binary model is trained on 41 development images. It is not an
    independently validated production model. Low confidence abstains. A
    positive detection receives an advisory subtype from the legacy tree and
    always requires technician confirmation.
    """
    import numpy as np

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if len(bgr.shape) == 3 else bgr
    if float(gray.std()) < 5.0 or float(gray.mean()) < 8.0:
        return {
            "damage_presence": {"value": None, "confidence": 0.0, "source": "random-forest-presence"},
            "damage_type": {"value": None, "confidence": 0.0, "source": "pending"},
            "raw_class": None,
            "status": "uncertain",
            "review_required": True,
            "retake_prompt": "Image is too dark or visually uniform to assess. Capture the affected surface with more light and texture visible.",
            "model_details": {"reason": "low_image_information"},
        }

    model_path = pathlib.Path(__file__).parent.parent / "models" / "damage_presence_rf.pkl"
    report_path = pathlib.Path(__file__).parent.parent / "eval" / "damage_presence_report.json"
    if not model_path.exists():
        return {
            "damage_presence": {"value": None, "confidence": 0.0, "source": "unavailable"},
            "damage_type": {"value": None, "confidence": 0.0, "source": "pending"},
            "raw_class": None,
            "status": "uncertain",
            "review_required": True,
            "retake_prompt": "The trained damage model is missing. Keep this capture for manual review.",
            "model_details": {"reason": "model_checkpoint_missing"},
        }

    with model_path.open("rb") as f:
        presence_model = pickle.load(f)
    probs = presence_model.predict_proba(np.asarray([damage.features(bgr)], dtype=float))[0]
    classes = list(presence_model.classes_)
    order = np.argsort(probs)[::-1]
    best_idx = int(order[0])
    presence = str(classes[best_idx])
    confidence = float(probs[best_idx])
    margin = float(probs[order[0]] - probs[order[1]]) if len(order) > 1 else confidence
    # Asymmetric thresholds: a false "no damage" can silently omit a claim
    # item, so negative decisions require stronger evidence than positive flags.
    class_threshold = confidence_threshold if presence == "damage" else no_damage_threshold
    accepted = confidence >= class_threshold and margin >= margin_threshold
    presence_field = {
        "value": presence if accepted else None,
        "confidence": round(confidence, 3),
        "source": "random-forest-presence",
        "decision_threshold": class_threshold,
    }
    details = {
        "model": "random_forest_damage_presence",
        "training_samples": 41,
        "damage_threshold": confidence_threshold,
        "no_damage_threshold": no_damage_threshold,
        "margin_threshold": margin_threshold,
        "decision_policy": "asymmetric_conservative_abstention",
        "probabilities": {str(classes[i]): round(float(probs[i]), 4) for i in range(len(classes))},
        "margin": round(margin, 4),
    }
    if presence_field["value"] is None:
        return {
            "damage_presence": presence_field,
            "damage_type": {"value": None, "confidence": 0.0, "source": "pending"},
            "raw_class": None,
            "status": "uncertain",
            "review_required": True,
            "retake_prompt": "Damage presence is uncertain. Capture the affected surface closer, with better lighting and less obstruction.",
            "model_details": details,
        }
    if presence == "none":
        return {
            "damage_presence": presence_field,
            "damage_type": {"value": None, "confidence": confidence, "source": "random-forest-presence"},
            "raw_class": "none",
            "status": "no_damage",
            "review_required": False,
            "retake_prompt": None,
            "model_details": details,
        }

    # Subtype is advisory only. The 4 class tree is not the trained presence model.
    with _TREE.open("rb") as f:
        subtype_model = pickle.load(f)
    subtype = str(subtype_model.predict(np.asarray([damage.features(bgr)]))[0])
    with _RESULTS.open(encoding="utf-8") as f:
        precision = json.load(f).get("precision_by_class", {})
    details["subtype_model"] = "legacy_cv_tree_advisory"
    return {
        "damage_presence": presence_field,
        "damage_type": {"value": TYPE_MAP.get(subtype), "confidence": round(float(precision.get(subtype, 0.0)), 3), "source": "legacy-tree-advisory"},
        "raw_class": subtype,
        "status": "review_required",
        "review_required": True,
        "review_reason": "The trained model detects damage presence only. The displayed subtype is an unvalidated advisory suggestion and must be confirmed by a technician.",
        "retake_prompt": None,
        "model_details": details,
    }
