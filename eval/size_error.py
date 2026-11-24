"""Measure size-estimate error against YOUR tape measurements. Usage: python eval/size_error.py tape.csv <images-dir> [assumed_dist_ft]
tape.csv columns: image,damage_class,tape_length_ft,tape_height_ft,distance_ft   (class: stain|crack|standing_water)
Reports mean absolute error per axis and area. Without your CSV there is NO size accuracy claim."""
import csv, os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fieldcapture import damage, lineitems
rows = list(csv.DictReader(open(sys.argv[1]))); root = sys.argv[2]; err_l, err_h, err_a, miss = [], [], [], 0
for r in rows:
    img = cv2.imread(os.path.join(root, r["image"])); bb = damage.region_bbox(img, r["damage_class"])
    if not bb: miss += 1; continue
    d = float(r.get("distance_ft") or (sys.argv[3] if len(sys.argv) > 3 else 4.0))
    w, h, _ = lineitems.estimate_size_without_depth(img.shape[1], img.shape[0], bb[2], bb[3], assumed_dist_ft=d)
    tl, th = float(r["tape_length_ft"]), float(r["tape_height_ft"])
    err_l.append(abs(w - tl)); err_h.append(abs(h - th)); err_a.append(abs(w * h - tl * th) / (tl * th))
m = lambda x: round(sum(x) / len(x), 2) if x else None
print({"n": len(rows), "no_region_found": miss, "mae_length_ft": m(err_l), "mae_height_ft": m(err_h), "mean_area_rel_error": m(err_a)})
