"""Inject real aggregated results into site/index.html, replacing the DATA/META block."""
import json, re

agg = json.load(open("data/our_results.json"))
A, meta = agg["aggregate"], agg["meta"]

MODELS = ["elman", "lstm", "gru", "transformer", "token_merge", "rlt"]
OPS = [16, 32, 64, 128]

def get(key, op):
    d = A.get(key)
    return round(d["mean"][str(op)], 1) if d else 0.0

results = {}
for task in ("parity", "five"):
    for m in MODELS:
        results[f"{task}|{m}"] = {op: get(f"{task}|{m}", op) for op in OPS}

PARAMS = {m: meta[m]["params"] for m in MODELS}
MS = {m: round(meta[m]["train_ms_per_step"]) for m in MODELS}

lines = []
lines.append("const DATA={tasks:{parity:{chance:50,label:\"Parity (chance 50%)\"},five:{chance:20,label:\"Five-state (chance 20%)\"}},")
lines.append(" results:{")
rows = []
for m in MODELS:
    rows.append("  " + ", ".join(
        f'"{task}|{m}":{{{", ".join(f"{o}:{results[f'{task}|{m}'][o]}" for o in OPS)}}}'
        for task in ("parity", "five")))
lines.append(",\n".join(rows))
lines.append("}};")
lines.append("const META={" + ", ".join(f'{m}:{{p:{PARAMS[m]},ms:{MS[m]}}}' for m in MODELS) + "};")
block = "\n".join(lines)

html = open("site/index.html").read()
pat = re.compile(r"const DATA=\{.*?const PUBLISHED=\{.*?\};", re.S)
assert pat.search(html), "DATA..PUBLISHED block not found"
pub = re.search(r"const PUBLISHED=\{.*?\};", html, re.S).group(0)
html = pat.sub(lambda _: block + "\n" + pub, html, count=1)
open("site/index.html", "w").write(html)
print("injected real results:")
for m in MODELS:
    print(f"  {m:12s} p={PARAMS[m]:6d} ms={MS[m]:4d} parity=",
          {o: results[f'parity|{m}'][o] for o in OPS},
          " five=", {o: results[f'five|{m}'][o] for o in OPS})
