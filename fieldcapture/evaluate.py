"""Evaluation scorer for JSONL ground-truth vs prediction rows."""
import json, re, sys
from collections import defaultdict


def norm(v):
    return None if v is None else re.sub(r"[^a-z0-9]", "", str(v).lower())


def score(rows):
    per = defaultdict(lambda: {"n": 0, "ok": 0, "conf_ok": [], "conf_bad": []})
    fails = []
    for r in rows:
        for k, tv in r.get("truth", {}).items():
            p = r.get("pred", {}).get(k) or {"value": None, "confidence": 0.0}
            ok = norm(p["value"]) == norm(tv)
            s = per[(r["kind"], k)]
            s["n"] += 1
            s["ok"] += ok
            (s["conf_ok"] if ok else s["conf_bad"]).append(p["confidence"])
            if not ok:
                fails.append({
                    "frame": r["frame"],
                    "kind": r["kind"],
                    "field": k,
                    "truth": tv,
                    "pred": p["value"],
                    "confidence": p["confidence"],
                    "lowlight": r.get("lowlight", False),
                })
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    table = {
        f"{a}.{b}": {
            "n": s["n"],
            "accuracy": round(s["ok"] / s["n"], 3),
            "mean_conf_when_right": mean(s["conf_ok"]),
            "mean_conf_when_wrong": mean(s["conf_bad"]),
        }
        for (a, b), s in sorted(per.items())
    }
    fails.sort(key=lambda f: -f["confidence"])
    return {
        "per_field": table,
        "worst_5": fails[:5],
        "frames": len(rows),
        "nameplates": sum(r.get("kind") == "nameplate" for r in rows),
        "lowlight": sum(bool(r.get("lowlight")) for r in rows),
    }


def requirements_met(rep):
    return rep["frames"] >= 40 and rep["nameplates"] >= 10 and rep["lowlight"] >= 10


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
    rep = score(rows)
    rep["meets_brief_minimums"] = requirements_met(rep)
    print(json.dumps(rep, indent=2))
