"""Regenerate site data (DATA/META constants) from results/*.json. Run after sweeps."""
import glob, json, re

MODELS = ["elman", "lstm", "gru", "transformer", "token_merge", "rlt"]
OPSES = [16, 32, 64, 128]

agg, meta = {}, {}
for f in glob.glob("results/*.json"):
    r = json.load(open(f))
    agg.setdefault((r["task"], r["model"]), {})[r["seed"]] = r

import statistics
res = {}
for m in MODELS:
    ms = []
    for t in ["parity", "five"]:
        for o in OPSES:
            vals = [agg[(t, m)][s]["acc"][str(o)] for s in agg[(t, m)]]
            res[f"{t}|{m}:{o}"] = round(statistics.mean(vals), 1)
    for s in agg[("parity", m)]:
        ms.append(agg[("parity", m)][s]["train_ms_per_step"])
    meta[m] = (agg[("parity", m)][0]["params"], round(statistics.mean(ms)))

lines = []
lines.append("const DATA={tasks:{parity:{chance:50,label:\"Parity (chance 50%)\"},five:{chance:20,label:\"Five-state (chance 20%)\"}},")
lines.append(" results:{")
pairs = []
for t in ["parity", "five"]:
    for m in MODELS:
        d = ",".join(f"{o}:{res[f'{t}|{m}:{o}']}" for o in OPSES)
        pairs.append(f'"{t}|{m}":{{{d}}}')
lines.append(" " + ",".join(pairs) + "}};")
lines.append("const META={" + ",".join(f'{m}:{{p:{meta[m][0]},ms:{meta[m][1]}}}' for m in MODELS) + "};")
block = "\n".join(lines)

html = open("site/index.html").read()
html = re.sub(r"const DATA=\{.*?const META=\{[^}]*\};", block.replace("\\", "\\\\"), html, flags=re.S)
open("site/index.html", "w").write(html)
print("site data regenerated from", len(glob.glob('results/*.json')), "runs")
print(block[:400])
