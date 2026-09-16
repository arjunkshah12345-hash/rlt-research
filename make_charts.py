"""Aggregate results and produce minimalist bar charts."""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["elman", "lstm", "gru", "transformer", "token_merge", "rlt"]
LABELS = {
    "elman": "Elman RNN\n(1990)", "lstm": "LSTM\n(1997)", "gru": "GRU\n(2014)",
    "transformer": "Transformer\n(2017)", "token_merge": "Token-merge\n(RLT ablation)",
    "rlt": "RLT-style\n(2026)",
}
COLORS = {
    "elman": "#c7cdd6", "lstm": "#aeb6c2", "gru": "#949eae",
    "transformer": "#6e7a8c", "token_merge": "#e0a458", "rlt": "#2f6feb",
}
TASKS = {"parity": "Parity (chance 50%)", "five": "Five-state tracking (chance 20%)"}
CHANCE = {"parity": 50, "five": 20}
OPSES = [16, 32, 64, 128]
BAR_SHADES = ["#9db4d0", "#7591b5", "#4c6f9e", "#26456e"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d1d5db", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#eef0f3", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titlelocation": "left",
})

def aggregate():
    agg, meta = {}, {}
    for f in glob.glob("results/*.json"):
        r = json.load(open(f))
        agg.setdefault((r["task"], r["model"]), []).append(r)
    out = {}
    for (task, model), runs in agg.items():
        accs = {str(ops): [run["acc"][str(ops)] for run in runs] for ops in OPSES}
        out[f"{task}|{model}"] = {
            "n_seeds": len(runs),
            "mean": {ops: float(np.mean(v)) for ops, v in accs.items()},
            "min": {ops: float(np.min(v)) for ops, v in accs.items()},
            "max": {ops: float(np.max(v)) for ops, v in accs.items()},
        }
    for m in MODELS:
        ms, params = [], None
        for f in glob.glob(f"results/{m}_*_seed*.json"):
            r = json.load(open(f))
            ms.append(r["train_ms_per_step"])
            params = r["params"]
        if ms:
            meta[m] = {"params": params, "train_ms_per_step": float(np.mean(ms))}
    return out, meta

def grouped_bars(ax, agg, task):
    width = 0.19
    x = np.arange(len(MODELS))
    for j, ops in enumerate(OPSES):
        means = [agg.get(f"{task}|{m}", {}).get("mean", {}).get(str(ops), np.nan) for m in MODELS]
        ax.bar(x + (j - 1.5) * width, means, width * 0.92, label=f"{ops} ops", color=BAR_SHADES[j])
    ax.axhline(CHANCE[task], color="#e5484d", lw=1, ls=(0, (4, 3)))
    ax.text(len(MODELS) - 0.55, CHANCE[task] + 1.5, "chance", color="#e5484d", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=8.5)
    ax.set_ylabel("Final-state accuracy (%)")
    ax.set_title(TASKS[task], fontsize=11, fontweight="bold", pad=10)

def main():
    agg, meta = aggregate()
    os.makedirs("data", exist_ok=True)
    json.dump({"aggregate": agg, "meta": meta}, open("data/our_results.json", "w"), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    for ax, task in zip(axes, ["parity", "five"]):
        grouped_bars(ax, agg, task)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig("charts/accuracy_by_length.png", bbox_inches="tight")

    # length-generalization curves with seed min/max bands
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    for ax, task in zip(axes, ["parity", "five"]):
        for m in MODELS:
            d = agg.get(f"{task}|{m}")
            if not d:
                continue
            ys = [d["mean"][str(o)] for o in OPSES]
            ax.plot(OPSES, ys, marker="o", ms=4, lw=1.6, color=COLORS[m], label=LABELS[m].replace("\n", " "))
            ax.fill_between(OPSES, [d["min"][str(o)] for o in OPSES], [d["max"][str(o)] for o in OPSES],
                            color=COLORS[m], alpha=0.12, lw=0)
        ax.axhline(CHANCE[task], color="#e5484d", lw=1, ls=(0, (4, 3)))
        ax.set_xscale("log", base=2)
        ax.set_xticks(OPSES); ax.set_xticklabels([str(o) for o in OPSES])
        ax.set_xlabel("Evaluation length (operations)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(TASKS[task], fontsize=11, fontweight="bold", pad=10)
        ax.set_ylim(0, 105)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("charts/length_generalization.png", bbox_inches="tight")

    # cost of recurrence
    avail = [m for m in MODELS if m in meta]
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
    ms = [meta[m]["train_ms_per_step"] for m in avail]
    bars = ax.bar([LABELS[m].replace("\n", " ") for m in avail], ms,
                  color=[COLORS[m] for m in avail], width=0.62)
    for b, v in zip(bars, ms):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}ms", ha="center", fontsize=8.5, color="#374151")
    ax.set_ylabel("Training step time (ms, batch 64)")
    ax.set_title("The hardware price of recurrence (CPU, per training step)", fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig("charts/cost_of_recurrence.png", bbox_inches="tight")

    # analytic temporal depth (RLT paper reference config LE=LD=48)
    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
    t = np.arange(1, 9)
    ax.plot(t, t * 48, lw=2, color="#2f6feb", label="RLT recurrent path (48·t blocks)")
    ax.plot(t, np.full_like(t, 96), lw=2, color="#6e7a8c", label="Standard deep Transformer (fixed 96 blocks)")
    ax.fill_between(t, 96, t * 48, color="#2f6feb", alpha=0.08, lw=0)
    ax.set_xlabel("Processed tokens t")
    ax.set_ylabel("Decoder blocks on the state path")
    ax.set_title("Temporal depth grows with the sequence (analytic, L=48)", fontsize=11, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig("charts/temporal_depth.png", bbox_inches="tight")

    print("charts written; run-groups aggregated:", len(agg))

if __name__ == "__main__":
    main()
