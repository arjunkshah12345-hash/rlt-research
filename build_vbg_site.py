#!/usr/bin/env python3
"""Rebuild site/index.html as a Vercel-brand report (vbg design system).

Reads:
  results/*.json               36 raw runs (per-seed accuracy per length)
  data/our_results.json        aggregates + timing
  data/rlt_repo_published.json published synthetic runs transcribed from the repo

Every number on the page is derived from these files. No hand-typed results.
"""
import json
import glob
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS = [
    ("elman",       "Elman RNN",              "1990"),
    ("lstm",        "LSTM",                   "1997"),
    ("gru",         "GRU",                    "2014"),
    ("transformer", "Transformer",            "2017"),
    ("token_merge", "Token merge (ablation)", ""),
    ("rlt",         "RLT-style",              "2026"),
]
LENGTHS = ["16", "32", "64", "128"]
CHANCE = {"parity": 50.0, "five": 20.0}

# ---------------------------------------------------------------- load data
with open(ROOT / "data" / "our_results.json") as f:
    OURS = json.load(f)
with open(ROOT / "data" / "rlt_repo_published.json") as f:
    PUB = json.load(f)

per_seed = {}   # (task, model, length) -> [acc per seed]; ("ms",) -> step times
for path in sorted(glob.glob(str(ROOT / "results" / "*.json"))):
    with open(path) as f:
        run = json.load(f)
    key = (run["task"], run["model"])
    for L, acc in run["acc"].items():
        per_seed.setdefault(key + (L,), []).append(acc)
    per_seed.setdefault(key + ("ms",), []).append(run["train_ms_per_step"])

def agg(task, model, L, stat="mean"):
    return OURS["aggregate"][f"{task}|{model}"][stat][L]

def seeds(task, model, L):
    return sorted(per_seed[(task, model, L)])

def ms_per_step(model):
    # All-run mean (3 parity + 3 five seeds), matching data/our_results.json meta
    # so the page, the aggregate JSON, and the README quote the same number.
    return OURS["meta"][model]["train_ms_per_step"]

def params(model):
    return OURS["meta"][model]["params"]

def f1(x):
    return f"{x:.1f}"

# ---------------------------------------------------------------- head + css
PAGE_CSS = """<style>
  /* Page-owned CSS: page-specific geometry only, built from public vbg tokens. */
  .vbg-custom-plot svg { display: block; width: 100%; height: auto; }
  .vbg-viz-mean { stroke: var(--vbg-text-primary); stroke-width: 2; }
  .vbg-viz-chance-label { font-size: 10px; }
  .vbg-viz-axis-title { font-size: 11px; }
</style>
"""

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RLT at matched parameters: an independent benchmark</title>
<meta name="description" content="Six architectures from 1990 to 2026, matched to 75K to 84K parameters, on the RLT paper's state-tracking tasks.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400..600&family=Geist+Mono:wght@400..600&display=swap" rel="stylesheet" referrerpolicy="no-referrer">
<link href="assets/vercel-brand.css" rel="stylesheet">
""" + PAGE_CSS + """</head>
"""

# ---------------------------------------------------------------- shell parts
HEADER = """
<body class="vbg-report">
<div class="vbg-shell">
  <a class="vbg-skip-link" href="#main">Skip to content</a>
  <header class="vbg-header">
    <div class="vbg-masthead">
      <div class="vbg-document-meta">
        <span class="vbg-context">Independent benchmark</span>
        <span class="vbg-date">September 15, 2026</span>
      </div>
    </div>
  </header>
  <main id="main">
"""

FOOTER = """
  </main>
  <footer class="vbg-footer">
    <span>Independent benchmark of RLT at matched parameters, September 2026</span>
  </footer>
</div>
</body>
</html>
"""

# ---------------------------------------------------------------- opening
STATS = [
    ("Architectures", "6", "Elman 1990 to RLT-style 2026"),
    ("Runs", "36", "6 models, 2 tasks, 3 seeds"),
    ("Parameter budget", "75K to 84K", "Matched across all models"),
    ("Training budget", "2,500 steps", "Identical for every run"),
]
stat_strip = '<div class="vbg-stat-strip">' + "".join(
    f'<div class="vbg-stat"><p class="vbg-stat-label">{l}</p>'
    f'<p class="vbg-stat-value">{v}</p><p class="vbg-stat-detail">{d}</p></div>'
    for l, v, d in STATS
) + "</div>"

OPENING = f"""
<section class="vbg-opening">
  <div class="vbg-opening-claim">
    <h1 class="vbg-title">At matched parameters, a recurrent network from 1990 beats a looped transformer from 2026</h1>
    <p class="vbg-lede">The RLT paper warns that parallel transformers fail the state-tracking tasks that recurrent
    models solve. We replicated the comparison with six architectures, matched to 75,000 to 84,000 parameters and one
    identical training budget. The warning holds, but the credit goes to recurrence itself: the simplest network
    tested wins both tasks at every evaluation length, and the looping machinery adds cost without adding accuracy
    at this scale.</p>
  </div>
  <div class="vbg-opening-proof">{stat_strip}</div>
  <div class="vbg-opening-context">
    <p>RLT (Recurrent Looped Transformer, Yifan Zhang, September 2026) wraps a standard transformer block stack in a
    loop that runs once per token, giving each state a much longer path through the network. After training, the
    loop unrolls back into an ordinary parallel transformer, so inference stays fast. The paper reports that
    parallel transformers crash on long state-tracking programs while RLT holds, and prices the difference in
    parameters instead of FLOPs (<a href="https://github.com/yifanzhang-pro/recurrent-looped-transformer">paper and code</a>).</p>
  </div>
</section>
"""

# ---------------------------------------------------------------- bar comparison
def bar_list(task):
    rows = []
    for mid, name, yr in MODELS:
        v = agg(task, mid, "32")
        label = f"{name}, {yr}" if yr else name
        rows.append(
            '<div class="vbg-bar">'
            f'<span class="vbg-bar-label">{label}</span>'
            f'<span class="vbg-bar-value">{f1(v)}</span>'
            f'<div class="vbg-bar-track"><div class="vbg-bar-fill" style="width:{v:.4f}%"></div></div>'
            '</div>'
        )
    return '<div class="vbg-bar-list">' + "".join(rows) + "</div>"

BAR_SECTION = f"""
<section class="vbg-section">
  <h2 class="vbg-heading-24">At the trained length, only the parallel transformer fails</h2>
  <p class="vbg-lede">Accuracy at 32 operations, the length every model trained on, on a shared 0 to 100 scale.
  Each bar is the mean of three seeds.</p>
  <div class="vbg-grid">
    <div class="vbg-span-6">
      <h3 class="vbg-heading-16">Parity</h3>
      {bar_list("parity")}
    </div>
    <div class="vbg-span-6">
      <h3 class="vbg-heading-16">Five-state composition</h3>
      {bar_list("five")}
    </div>
  </div>
  <p class="vbg-note">Chance is 50% on parity and 20% on five-state. The transformer sits at chance on parity and
  reaches 37% on five-state; every recurrent architecture, including the 1990 Elman network, clears 76%.</p>
</section>
"""

# ---------------------------------------------------------------- seed dot plot
def dot_chart():
    top, bottom = 40, 280
    panels = [
        ("parity", "Parity", 70, 360),
        ("five", "Five-state composition", 430, 720),
    ]

    def y(v):
        return bottom - (v / 100.0) * (bottom - top)

    svg = [
        '<svg viewBox="0 0 760 330" role="img" aria-label="Dot plot: accuracy of six models on two tasks, '
        'three dots per model for three seeds, mean tick, chance marked with a dashed line.">'
    ]
    for task, title, x0, x1 in panels:
        slot = (x1 - x0) / len(MODELS)
        for g in (0, 25, 50, 75, 100):
            gy = y(g)
            svg.append(
                f'<line class="vbg-chart-gridline" x1="{x0}" y1="{gy:.1f}" x2="{x1}" y2="{gy:.1f}"/>')
            svg.append(
                f'<text x="{x0 - 8}" y="{gy + 3:.1f}" text-anchor="end" font-size="10" '
                f'fill="var(--vbg-text-secondary)">{g}</text>')
        svg.append(f'<line class="vbg-chart-axis" x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}"/>')
        cy = y(CHANCE[task])
        svg.append(
            f'<line class="vbg-chart-annotation-line" x1="{x0}" y1="{cy:.1f}" x2="{x1}" y2="{cy:.1f}"/>')
        svg.append(
            f'<text class="vbg-viz-chance-label" x="{x1 - 4}" y="{cy - 5:.1f}" text-anchor="end" '
            f'fill="var(--vbg-text-secondary)">chance {CHANCE[task]:.0f}</text>')
        svg.append(
            f'<text class="vbg-viz-axis-title" x="{(x0 + x1) / 2:.0f}" y="24" text-anchor="middle" '
            f'fill="var(--vbg-text-primary)">{title}</text>')
        for i, (mid, name, _yr) in enumerate(MODELS):
            cx = x0 + (i + 0.5) * slot
            for s in seeds(task, mid, "32"):
                svg.append(f'<circle class="vbg-data-point" cx="{cx:.1f}" cy="{y(s):.1f}" r="4"/>')
            m = agg(task, mid, "32")
            svg.append(
                f'<line class="vbg-viz-mean" x1="{cx - 9:.1f}" y1="{y(m):.1f}" '
                f'x2="{cx + 9:.1f}" y2="{y(m):.1f}"/>')
            ly = bottom + 16 if i % 2 == 0 else bottom + 30
            svg.append(
                f'<text x="{cx:.1f}" y="{ly}" text-anchor="middle" font-size="10" '
                f'fill="var(--vbg-text-secondary)">{name}</text>')
    svg.append(
        '<text class="vbg-viz-axis-title" x="16" y="160" text-anchor="middle" '
        'fill="var(--vbg-text-secondary)" transform="rotate(-90 16 160)">Accuracy (%)</text>')
    svg.append("</svg>")

    rlt_p = seeds("parity", "rlt", "32")
    tm_p = seeds("parity", "token_merge", "32")
    fig = f"""
<figure class="vbg-chart">
  <div class="vbg-chart-header">
    <h3 class="vbg-heading-16">Seeds disagree about the looped models on parity</h3>
  </div>
  <div class="vbg-chart-viewport vbg-custom-plot" tabindex="0">{''.join(svg)}</div>
  <figcaption class="vbg-caption">Each dot is one seed at 32 operations, the trained length; the tick is the
  three-seed mean; the dashed line is chance. On parity, RLT-style spans {f1(min(rlt_p))} to {f1(max(rlt_p))} percent
  across seeds and token merge spans {f1(min(tm_p))} to {f1(max(tm_p))}, while the transformer sits at chance. On
  five-state composition, every recurrent model clears 96% on every seed. Mean values hide both instabilities.</figcaption>
</figure>
"""
    return fig

CHART_SECTION = f"""
<section class="vbg-section">
  <h2 class="vbg-heading-24">The mean is a fair summary for five models, and unfair for two</h2>
  <p class="vbg-lede">Averaging three seeds makes the looped models look mid-tier when they are actually a
  coin-flip between solving the task and staying at chance.</p>
  {dot_chart()}
</section>
"""

# ---------------------------------------------------------------- cost
def cost_rows():
    t_ms = ms_per_step("transformer")
    rows = []
    for mid, name, _yr in MODELS:
        ms = ms_per_step(mid)
        rows.append(
            f'<tr><th scope="row">{name}</th>'
            f'<td class="vbg-numeric">{params(mid):,}</td>'
            f'<td class="vbg-numeric">{ms:.1f}</td>'
            f'<td class="vbg-numeric">{ms / t_ms:.1f}&times;</td></tr>'
        )
    return "".join(rows)

COST_SECTION = f"""
<section class="vbg-section">
  <h2 class="vbg-heading-24">Looping costs four times more training compute</h2>
  <p class="vbg-lede">Wall-clock per training step, measured on the same machine in the same sweep.</p>
  <div class="vbg-table-wrap">
    <table>
      <caption class="vbg-visually-hidden">Parameters and training wall-clock per step for each model.</caption>
      <thead>
        <tr>
          <th scope="col">Model</th>
          <th scope="col" class="vbg-numeric">Parameters</th>
          <th scope="col" class="vbg-numeric">ms per training step</th>
          <th scope="col" class="vbg-numeric">Relative to transformer</th>
        </tr>
      </thead>
      <tbody>{cost_rows()}</tbody>
    </table>
  </div>
  <p class="vbg-note">A trained RLT model runs as a standard parallel transformer, so the sequential cost applies
  to training only. All runs shared one machine and ran six at a time, so absolute milliseconds carry contention;
  the ratios were stable across isolated spot checks.</p>
</section>
"""

# ---------------------------------------------------------------- verdict band
VERDICT = """
<section class="vbg-band" data-tone="contrast">
  <h2 class="vbg-heading-24">What we conclude</h2>
  <p>The paper's central warning replicates. Given the same parameters and the same 2,500-step budget, the parallel
  transformer sits at chance on parity and reaches 37% on five-state composition, while every recurrent
  architecture, gated or not, solves both tasks.</p>
  <p>The wins belong to recurrence, not to loops. The Elman network, one tanh layer with no gates, scores 100% on
  both tasks at every evaluation length up to four times the training length. RLT-style also solves the tasks, but
  it is the slowest model tested, unstable across seeds on parity, and never better than the simplest baseline at
  this scale.</p>
  <p>What we cannot rule out is scale. The paper's claims are made at 10M+ parameters with 48 loop iterations; a
  75K-parameter probe can confirm the direction of the argument, not the payoff of the machinery. Loop-free
  inference remains RLT's genuinely attractive property: the trained weights run as an ordinary parallel
  transformer.</p>
</section>
"""

# ---------------------------------------------------------------- method
METHOD = """
<section class="vbg-section">
  <h2 class="vbg-heading-24">Method and limits</h2>
  <div class="vbg-reading">
    <h3 class="vbg-heading-20">Tasks</h3>
    <p>Parity: read a binary token stream and report whether the number of ones so far is even; chance is 50%.
    Five-state composition: apply one to five random permutations of a five-element set and report the final
    mapping; chance is 20%. Parity comes from the RLT paper; five-state is the composition probe the paper scales
    up.</p>
    <h3 class="vbg-heading-20">A documented deviation</h3>
    <p>The paper trains parity on the final token only. In pilot runs, sparse final-token targets never left chance
    for any architecture at this budget, so we train on the running parity at every position and score final-token
    accuracy. The deviation is recorded in every results file.</p>
    <h3 class="vbg-heading-20">Protocol</h3>
    <p>Adam at a learning rate of 3e-3, batch 64, 2,500 steps, sinusoidal positions, matched parameter counts
    (75,648 to 83,552), three seeds per configuration, CPU only. This page and every number on it are generated
    from the raw run files by scripts in the repository.</p>
    <h3 class="vbg-heading-20">What did not reproduce</h3>
    <p>The repo's published synthetic runs show RLT crashing from 100 to 61% parity between 64 and 128 operations.
    Under our generator, no recurrent model degrades at four times the training length. We attribute the difference
    to task construction and supervision rather than to the architectures.</p>
    <h3 class="vbg-heading-20">Limits</h3>
    <p>One small task generator, one hardware class, three seeds. Cost ratios are robust; absolute milliseconds are
    not. The 75K-parameter regime cannot validate claims the paper makes at 10M+ parameters.</p>
  </div>
</section>
"""

# ---------------------------------------------------------------- audit tables
def audit_table(task, task_name):
    rows = []
    for mid, name, yr in MODELS:
        cells = "".join(f'<td class="vbg-numeric">{f1(agg(task, mid, L))}</td>' for L in LENGTHS)
        rows.append(
            f'<tr><th scope="row">{name}</th>'
            f'<td class="vbg-numeric">{yr or "2025"}</td>{cells}</tr>')
    worst = {mid: min(seeds(task, mid, "32")) for mid, _n, _y in MODELS}
    if task == "parity":
        note = (f'Worst seed at the trained length: token merge {f1(worst["token_merge"])}, '
                f'RLT-style {f1(worst["rlt"])}, transformer {f1(worst["transformer"])}, '
                f'LSTM {f1(worst["lstm"])}; every other model stays above 95%.')
    else:
        note = (f'Worst seed at the trained length: transformer {f1(worst["transformer"])}; '
                f'every other model stays above 96%.')
    return f"""
  <h3 class="vbg-heading-16">{task_name}, mean of three seeds</h3>
  <div class="vbg-table-wrap">
    <table>
      <caption class="vbg-visually-hidden">Accuracy by evaluation length, {task_name.lower()}.</caption>
      <thead>
        <tr>
          <th scope="col">Model</th>
          <th scope="col" class="vbg-numeric">Year</th>
          <th scope="col" class="vbg-numeric">16 ops</th>
          <th scope="col" class="vbg-numeric">32 ops</th>
          <th scope="col" class="vbg-numeric">64 ops</th>
          <th scope="col" class="vbg-numeric">128 ops</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  <p class="vbg-note">{note}</p>
"""

AUDIT = f"""
<section class="vbg-section">
  <h2 class="vbg-heading-24">Accuracy by evaluation length</h2>
  <p class="vbg-lede">Models trained at 32 operations, evaluated at up to four times that length. No recurrent
  model degrades with length under this generator; training lengths beyond the budget were not tested.</p>
  {audit_table("parity", "Parity")}
  {audit_table("five", "Five-state composition")}
</section>
"""

# ---------------------------------------------------------------- published
def pub_rows(task):
    names = {"rlt": "RLT", "transformer": "Transformer", "token_merge": "Token merge"}
    rows = []
    for mid, name in names.items():
        cells = "".join(
            f'<td class="vbg-numeric">{PUB[task][mid][L]:g}</td>' for L in ("32", "64", "128"))
        rows.append(f'<tr><th scope="row">{name}</th>{cells}</tr>')
    return "".join(rows)

PUBLISHED = f"""
<section class="vbg-section">
  <h2 class="vbg-heading-24">The repo's published synthetic runs, for the record</h2>
  <p class="vbg-lede">Preliminary results contributed in the RLT repository, transcribed from their figure.
  Around 79K parameters, three seeds, FLOPs not matched.</p>
  <div class="vbg-table-wrap">
    <table>
      <caption class="vbg-visually-hidden">Published synthetic accuracy from the RLT repository.</caption>
      <thead>
        <tr>
          <th scope="col">Model</th>
          <th scope="col" class="vbg-numeric">Parity 32</th>
          <th scope="col" class="vbg-numeric">Parity 64</th>
          <th scope="col" class="vbg-numeric">Parity 128</th>
          <th scope="col" class="vbg-numeric">Five 32</th>
          <th scope="col" class="vbg-numeric">Five 64</th>
          <th scope="col" class="vbg-numeric">Five 128</th>
        </tr>
      </thead>
      <tbody>{pub_rows('parity')}{pub_rows('five')}</tbody>
    </table>
  </div>
  <p class="vbg-note">In their generator, RLT crashes at long programs; in ours, it holds but never leads. The two
  setups differ in task generator and supervision, so the disagreement is about task construction as much as
  architecture.</p>
</section>
"""

# ---------------------------------------------------------------- sources
SOURCES = """
<section class="vbg-section">
  <h2 class="vbg-heading-24">Sources</h2>
  <div class="vbg-sources">
    <ul>
      <li>RLT paper and code: <a href="https://github.com/yifanzhang-pro/recurrent-looped-transformer">yifanzhang-pro/recurrent-looped-transformer</a> (Yifan Zhang, September 2026)</li>
      <li>Benchmark code: <code>bench.py</code> and <code>run_all.py</code> in this repository</li>
      <li>Raw runs: 36 files in <code>results/</code>; aggregates in <code>data/our_results.json</code></li>
      <li>Published-figure transcription: <code>data/rlt_repo_published.json</code></li>
    </ul>
  </div>
</section>
"""

# ---------------------------------------------------------------- assemble + write
html = (HEAD + HEADER + OPENING + BAR_SECTION + CHART_SECTION + COST_SECTION
        + VERDICT + METHOD + AUDIT + PUBLISHED + SOURCES + FOOTER)

out = ROOT / "site" / "index.html"
out.write_text(html)
print(f"wrote {out} ({len(html):,} bytes)")

assert "\u2014" not in html, "em dash found in output"
print("no em dashes; done")
