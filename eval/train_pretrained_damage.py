#!/usr/bin/env python3
"""Property separated damage model training and evaluation.

Manifest JSONL schema, one row per image:
{"image":"images/frame001.jpg","label":"crack","property_id":"property_001"}
property_id must identify the real physical property, not the archive/folder name.
"""
import argparse, json
from pathlib import Path
from fieldcapture.pretrained_damage import evaluate_grouped, train_final

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--epochs",type=int,default=12)
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--output",default="models/damage_resnet18.pt")
    ap.add_argument("--evaluate-only",action="store_true")
    args=ap.parse_args()
    result=evaluate_grouped(args.manifest,epochs=args.epochs,batch_size=args.batch_size)
    Path("eval").mkdir(exist_ok=True)
    Path("eval/damage_resnet18_property_grouped.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not args.evaluate_only:
        print(json.dumps(train_final(args.manifest,args.output,epochs=args.epochs,batch_size=args.batch_size),indent=2))
if __name__=="__main__": main()
