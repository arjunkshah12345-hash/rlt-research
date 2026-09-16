"""Parallel driver for the full 6-model x 2-task x 3-seed sweep."""
import itertools, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

MODELS = ["elman", "lstm", "gru", "transformer", "token_merge", "rlt"]
TASKS = ["parity", "five"]
SEEDS = [0, 1, 2]
STEPS = 2500

jobs = list(itertools.product(TASKS, MODELS, SEEDS))

def run(job):
    task, model, seed = job
    out = os.path.join("results", f"{model}_{task}_seed{seed}.json")
    if os.path.exists(out):
        print(f"skip {out}", flush=True)
        return
    log = open(f"logs/{model}_{task}_s{seed}.log", "w")
    t0 = time.time()
    subprocess.run(
        [sys.executable, "bench.py", "--model", model, "--task", task,
         "--seed", str(seed), "--steps", str(STEPS), "--out", "results"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    print(f"done {model}/{task}/s{seed} in {time.time()-t0:.0f}s", flush=True)

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
with ThreadPoolExecutor(max_workers=6) as ex:
    ex.map(run, jobs)
print("ALL DONE", flush=True)
