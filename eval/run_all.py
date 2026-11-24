"""Regenerates eval/REPORT.md and all numbers from the images. Fully offline.
Usage: python eval/run_all.py <root-dir-with-extracted-image-folders> [--ocr]    (--ocr re-runs Tesseract, ~4 min)"""
import collections, glob, json, os, sys, time, pickle, warnings
import cv2, numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from fieldcapture import damage, evaluate, extract, process, quality

root = sys.argv[1]; RUN_OCR = "--ocr" in sys.argv
os.chdir(root)
fs = [f for f in sorted(glob.glob("*/**/*.jp*g", recursive=True)) if "fieldcapture-complete" not in f]
assert len(fs) == 51, len(fs)
img = lambda i: cv2.imread(os.path.join(root, fs[i]))
GROUP = lambda i: "nameplate" if i < 10 else "extra" if i < 21 else "home" if i < 31 else "lowlight" if i < 41 else "wall"
R = {}

# ---------------- quality gate calibration ----------------
retake = {31, 32, 33, 38, 40}                       # my usability labels: too dark / glare to read detail
good = [quality.blur_ratio(img(i)) for i in range(51)]
blur = {s: [quality.blur_ratio(cv2.GaussianBlur(img(i), (0, 0), s)) for i in range(51)] for s in (1.5, 3.0)}
cut = quality.V2["max_blur_ratio"]
luma = {i: float(cv2.cvtColor(img(i), cv2.COLOR_BGR2GRAY).mean()) for i in range(51)}
lcut = quality.V2["min_luma"]
fr_good = [quality.framing_ratio(cv2.cvtColor(img(i), cv2.COLOR_BGR2GRAY)) for i in range(10)]
def corner(i):
    im = img(i); h, w = im.shape[:2]; return cv2.cvtColor(im[int(h * .5):, int(w * .5):], cv2.COLOR_BGR2GRAY)
fr_bad = [quality.framing_ratio(corner(i)) for i in range(10)]
v2 = {i: quality.check_v2(img(i), "nameplate" if i < 10 else "damage") for i in range(51)}
R["quality"] = {
    "blur_good_range": [round(min(good), 2), round(max(good), 2)],
    "blur_false_rejects_at_cut": int(sum(x > cut for x in good)),
    "blur_catch_sigma1.5": round(float(np.mean([x > cut for x in blur[1.5]])), 3),
    "blur_catch_sigma3.0": round(float(np.mean([x > cut for x in blur[3.0]])), 3),
    "luma_retake_values": sorted(round(luma[i], 1) for i in retake),
    "luma_usable_min": round(min(luma[i] for i in range(51) if i not in retake), 1),
    "luma_cut_label_accuracy": round(float(np.mean([(luma[i] < lcut) == (i in retake) for i in range(51)])), 3),
    "framing_good_range": [round(min(fr_good), 2), round(max(fr_good), 2)],
    "framing_catch_corner_crop": round(float(np.mean([x < quality.V2["min_framing"] for x in fr_bad])), 2),
    "framing_false_rejects_nameplates": int(sum(x < quality.V2["min_framing"] for x in fr_good)),
    "v1_gate_rejects_of_10_wall_photos": int(sum(not quality.check(cv2.resize(cv2.cvtColor(img(i), cv2.COLOR_BGR2GRAY), (160, 120)))["pass"] for i in range(41, 51))),
    "v2_gate_pass_by_group": {g: f"{sum(v2[i]['pass'] for i in range(51) if GROUP(i) == g)}/{sum(GROUP(i) == g for i in range(51))}" for g in ("nameplate", "extra", "home", "lowlight", "wall")},
    "v2_rejected": {i: v2[i]["reasons"] for i in range(51) if not v2[i]["pass"]},
}

# ---------------- nameplate OCR eval ----------------
lab = json.load(open(os.path.join(HERE, "labels_nameplates.json")))
np_res = {}
for mode in ("full", "fast"):
    rp = os.path.join(HERE, f"nameplate_rows_{mode}.json")
    if RUN_OCR or not os.path.exists(rp):
        rows = []; t = time.time()
        for i in range(10):
            rows.append({"frame": f"{os.path.basename(fs[i])}#{i}", "kind": "nameplate", "lowlight": False, "truth": lab[str(i)], "pred": extract.extract_nameplate(os.path.join(root, fs[i]), fast=(mode == "fast"))})
        json.dump(rows, open(rp, "w")); json.dump({"secs_per_plate": (time.time() - t) / 10}, open(rp + ".time", "w"))
    rows = json.load(open(rp)); rep = evaluate.score(rows)
    t = json.load(open(rp + ".time"))["secs_per_plate"] if os.path.exists(rp + ".time") else None
    np_res[mode] = {"report": rep, "secs": t}
R["nameplate"] = np_res

# ---------------- damage classifier: leave-one-out ----------------
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
dl = json.load(open(os.path.join(HERE, "damage_labels.json"))); idx = sorted(int(k) for k in dl)
X = np.array([damage.features(img(i)) for i in idx]); y = np.array([dl[str(i)] for i in idx])
pred, conf = [], []
for tr, te in LeaveOneOut().split(X):
    m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=0).fit(X[tr], y[tr]); pr = m.predict_proba(X[te])[0]
    pred.append(str(m.classes_[pr.argmax()])); conf.append(float(pr.max()))
pred = np.array(pred); conf = np.array(conf)
prec = {c: round(float(np.mean(y[pred == c] == c)), 3) for c in sorted(set(pred))}
final = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=0).fit(X, y); pickle.dump(final, open(os.path.join(HERE, "damage_tree.pkl"), "wb"))
cm = collections.defaultdict(collections.Counter)
for t_, p_ in zip(y, pred): cm[t_][p_] += 1
R["damage"] = {"n": len(y), "classes": dict(collections.Counter(y.tolist())), "baseline": round(max(collections.Counter(y.tolist()).values()) / len(y), 3),
               "loo_accuracy": round(float((pred == y).mean()), 3), "tree_conf_right": round(float(conf[pred == y].mean()), 3), "tree_conf_wrong": round(float(conf[pred != y].mean()), 3),
               "precision_by_class": prec, "confusion": {k: dict(v) for k, v in cm.items()}, "errors": [(int(i), t_, p_) for i, t_, p_ in zip(idx, y, pred) if t_ != p_]}
json.dump({**R["damage"]}, open(os.path.join(HERE, "damage_results.json"), "w"), indent=1)

# ---------------- sample job record from the uploaded stills ----------------
items = [(2, "capture nameplate"), (15, "capture damage kitchen ceiling"), (18, "capture damage bedroom north wall"), (13, "capture damage basement floor")]
caps, log = [], []
for n, (i, cmd) in enumerate(items, 1):
    r = process.process_capture(os.path.join(root, fs[i]), cmd, f"c{n}", "2026-11-23T12:00:00Z", fast_ocr=True)
    log.append({"image": os.path.basename(fs[i]), "command": cmd, "accepted": r["accepted"], "retake_prompt": r.get("retake_prompt")})
    if r["accepted"]: caps.append(r["capture"])
note = __import__("fieldcapture.commands", fromlist=["x"]).parse_note_fields("subfloor soft by the cabinet in the kitchen")
caps.append({"id": "c5", "type": "note", "ts": "2026-11-23T12:01:00Z", "sync": "queued", "quality": None, "fields": note})
rec = process.build_record("JOB-SAMPLE-1", "2026-11-23T12:00:00Z", caps)
rec["notes"] = "Built from uploaded still images with typed commands. NOT a live walk-through. damage sizes left empty (no depth sensor, none estimated)."
json.dump(rec, open(os.path.join(HERE, "sample_job_record.json"), "w"), indent=1); R["sample"] = {"log": log, "line_items": rec["line_items"]}
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ("quality", "damage")}, indent=1, default=str)); print(json.dumps(R["sample"], indent=1, default=str))
for m in ("full", "fast"): print(m, np_res[m]["secs"], {k: v["accuracy"] for k, v in np_res[m]["report"]["per_field"].items()})
