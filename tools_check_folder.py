"""Run the quality gate over a folder of images. Usage: python tools_check_folder.py dev_samples   (needs numpy, pillow)"""
import sys, pathlib, numpy as np
from PIL import Image
from fieldcapture import quality

for p in sorted(pathlib.Path(sys.argv[1]).glob("*.jpg")):
    g = np.asarray(Image.open(p).convert("L").resize((160, 120)), dtype=float)
    r = quality.check(g); m = r["metrics"]
    print(f"{p.name:24} sharp {m['sharpness']:8.1f} light {m['luma']:6.1f} frame {m['framing']:5.2f}  {'PASS' if r['pass'] else 'FAIL: ' + '; '.join(r['reasons'])}")
