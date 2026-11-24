"""Checks that the JS quality metrics in web_prototype.html match fieldcapture.quality on the same arrays."""
import json, os, re, shutil, subprocess, sys, tempfile, unittest
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fieldcapture import quality


@unittest.skipUnless(shutil.which("node"), "node not installed")
class Parity(unittest.TestCase):
    def test_js_matches_python(self):
        html = open(os.path.join(ROOT, "web_prototype.html"), encoding="utf-8").read()
        fn = re.search(r"function metricsGray\(.*?\n(?=function measure)", html, re.S).group(0)
        r = np.random.default_rng(7)
        arrays = [r.random((120, 160)) * 255, r.random((120, 160)) * 30, np.full((120, 160), 90.0)]
        g = np.zeros((120, 160)); g[:, :30] = r.random((120, 30)) * 255; arrays.append(g)
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "a.json"), "w") as fh: json.dump([a.ravel().tolist() for a in arrays], fh)
        open(os.path.join(d, "t.js"), "w").write(fn + "\nconst A=require('fs');const arrs=JSON.parse(A.readFileSync(process.argv[2]));"
                                                 "console.log(JSON.stringify(arrs.map(a=>metricsGray(Float64Array.from(a),160,120))))")
        out = json.loads(subprocess.check_output(["node", os.path.join(d, "t.js"), os.path.join(d, "a.json")]))
        for a, js in zip(arrays, out):
            py = quality.metrics(a)
            self.assertAlmostEqual(js["blur"], py["sharpness"], delta=0.06 + 1e-4 * py["sharpness"])
            self.assertAlmostEqual(js["luma"], py["luma"], delta=0.06)
            self.assertAlmostEqual(js["framing"], py["framing"], delta=0.006)


if __name__ == "__main__": unittest.main()
