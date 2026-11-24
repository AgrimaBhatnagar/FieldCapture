import sys, json, time, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from fieldcapture import ocr, evaluate
fs = json.load(open('/tmp/files.json')); lab = json.load(open(f'{HERE}/labels_nameplates.json')); out = {}
import os as _o
prev = json.load(open(f"{HERE}/ocr_comparison.json")) if _o.path.exists(f"{HERE}/ocr_comparison.json") else {}
out.update(prev)
MODES = {"norefine": ("full_norefine", dict(fast=False, refine=False)), "refine": ("full_refine", dict(fast=False, refine=True)), "fastrefine": ("fast_refine", dict(fast=True, refine=True)), "fastnorefine": ("fast_norefine", dict(fast=True, refine=False))}
for name, kw in [MODES[sys.argv[1]]]:
    rows = []; t = time.time()
    for i in range(10):
        rows.append({"frame": f"{os.path.basename(fs[i])}#{i}", "kind": "nameplate", "lowlight": False, "truth": lab[str(i)], "pred": ocr.extract_nameplate('/tmp/up/' + fs[i], **kw)})
    rep = evaluate.score(rows); out[name] = {"secs_per_plate": round((time.time() - t) / 10, 1), "acc": {k: v["accuracy"] for k, v in rep["per_field"].items()},
        "errors": [(r["frame"], k, tv, r["pred"][k]["value"], r["pred"][k]["confidence"]) for r in rows for k, tv in r["truth"].items() if evaluate.norm(r["pred"][k]["value"]) != evaluate.norm(tv)]}
    json.dump(rows, open(f'{HERE}/nameplate_rows_{name}.json', 'w')); json.dump(out, open(f'{HERE}/ocr_comparison.json', 'w'), indent=1)
print("done")
