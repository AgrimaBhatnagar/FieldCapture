#!/usr/bin/env python3
"""Run Tesseract and PaddleOCR on the same nameplate image set and report differences."""
import argparse, json, time, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
from fieldcapture.ocr import _tesseract_words, _norm
from fieldcapture.ocr_backend import paddle_words

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("image_dir"); ap.add_argument("--output",default="eval/ocr_engine_comparison.json"); args=ap.parse_args()
    root=Path(args.image_dir); images=sorted(p for p in root.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
    if not images: raise SystemExit(f"No images found in {root}")
    report={"images":len(images),"engines":{"tesseract":{},"paddleocr":{}},"per_image":[],"note":"Agreement is not accuracy. Ground truth is required to score correctness."}
    totals={"tesseract":0.0,"paddleocr":0.0}; counts={"tesseract":0,"paddleocr":0}
    for path in images:
        im=cv2.imread(str(path))
        row={"image":path.name}
        for name,fn in (("tesseract",lambda:_tesseract_words(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),11)),("paddleocr",lambda:paddle_words(im))):
            t=time.perf_counter()
            try:
                words=fn(); elapsed=time.perf_counter()-t
                row[name]={"text":[w[0] for w in words],"mean_confidence":round(sum(w[1] for w in words)/len(words),2) if words else 0.0,"seconds":round(elapsed,3)}
                totals[name]+=elapsed; counts[name]+=1
            except Exception as exc: row[name]={"error":f"{type(exc).__name__}: {exc}"}
        a={_norm(x) for x in row.get("tesseract",{}).get("text",[])}; b={_norm(x) for x in row.get("paddleocr",{}).get("text",[])}
        row["token_set_jaccard"] = round(len(a&b)/len(a|b),3) if a|b else None
        report["per_image"].append(row)
    for name in totals:
        report["engines"][name]={"successful_images":counts[name],"mean_seconds":round(totals[name]/counts[name],3) if counts[name] else None}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
