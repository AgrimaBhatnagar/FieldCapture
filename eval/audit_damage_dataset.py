#!/usr/bin/env python3
"""Audit four class labels and property independence before training.

The manifest must be JSONL with image, label and real property_id. Do not use
lighting folders, capture sessions, or filenames as a substitute for property_id.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = {"crack", "stain", "standing_water", "none"}

def audit(path):
    p = Path(path)
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    errors=[]; by_property=defaultdict(list); labels=Counter(); sources=defaultdict(set)
    for n,row in enumerate(rows,1):
        for key in ("image","label","property_id"):
            if not row.get(key): errors.append(f"line {n}: missing {key}")
        if row.get("label") not in CLASSES: errors.append(f"line {n}: invalid label {row.get('label')!r}")
        if row.get("image"):
            img=Path(row["image"])
            if not img.is_absolute(): img=p.parent/img
            if not img.is_file(): errors.append(f"line {n}: image not found: {img}")
        if row.get("property_id"):
            by_property[str(row["property_id"])].append(row)
        if row.get("label") in CLASSES: labels[row["label"]]+=1
        if row.get("source_id") and row.get("property_id"):
            sources[str(row["source_id"])].add(str(row["property_id"]))
    for source, props in sources.items():
        if len(props)>1: errors.append(f"source_id {source!r} appears under multiple property IDs: {sorted(props)}")
    result={"images":len(rows),"properties":len(by_property),"class_counts":dict(labels),
            "property_image_counts":{k:len(v) for k,v in by_property.items()},
            "eligible_for_grouped_validation": not errors and len(by_property)>=3 and set(labels)==CLASSES,
            "errors":errors}
    if len(by_property)<3: result["errors"].append("Need at least 3 distinct real property IDs for leave one property out evaluation")
    if set(labels)!=CLASSES: result["errors"].append("Need examples of all four classes")
    result["eligible_for_grouped_validation"]=not result["errors"]
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("manifest"); args=ap.parse_args()
    result=audit(args.manifest); print(json.dumps(result,indent=2))
    if not result["eligible_for_grouped_validation"]: raise SystemExit(2)
