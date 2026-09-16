"""
RLT research benchmark: apples-to-apples state-tracking comparison across neural-net eras.
Models: elman, lstm, gru, transformer, token_merge, rlt
Tasks:  parity (chance 50%), five (chance 20%)
Protocol (mirrors the RLT repo's preliminary study):
  ~79K params, 3 seeds, train at 32 ops, eval at 16/32/64/128 ops, 2048 test programs per length.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- tasks ----------------

def gen_automaton(rng, n=5):
    # random permutation per transition token => T[s][a] gives next state
    return np.stack([rng.permutation(n) for _ in range(n)])

def make_batch(task, ops, bs, rng, auto):
    if task == "parity":
        bits = rng.integers(0, 2, size=(bs, ops))
        ans = bits.sum(axis=1) % 2
        seq = bits
    else:
        seq = rng.integers(0, 5, size=(bs, ops))
        state = np.zeros(bs, dtype=np.int64)
        for t in range(ops):
            state = auto[state, seq[:, t]]
        ans = state
    return torch.from_numpy(seq).long(), torch.from_numpy(ans).long()

def make_fixed_eval(task, ops, n, rng, auto):
    seqs, anss = [], []
    for _ in range(n):
        s, a = make_batch(task, ops, 1, rng, auto)
        seqs.append(s[0]); anss.append(a[0])
    return torch.stack(seqs), torch.stack(anss)

# ---------------- modules ----------------

class CausalAttn(nn.Module):
    """Multi-head causal self-attention with optional sliding window."""
    def __init__(self, d, h, window=None):
        super().__init__()
        self.d, self.h, self.w = d, h, window
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        if self.w is not None:
            keep = torch.arange(T, device=x.device)[:, None] >= (torch.arange(T, device=x.device)[None, :] - (self.w - 1))
            causal = causal & keep
        att = att.masked_fill(~causal, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(y)

class Block(nn.Module):
    def __init__(self, d, h, ffn, window=None):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalAttn(d, h, window)
        self.f1, self.f2 = nn.Linear(d, ffn, bias=False), nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.f2(F.gelu(self.f1(self.ln2(x))))

class CrossAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.d, self.h = d, h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

# ---------------- models ----------------

def sinusoidal(T, d, device):
    pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(0, d, 2, dtype=torch.float32, device=device)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang[:, : d // 2])
    return pe

class TransformerLM(nn.Module):
    def __init__(self, vocab, d=56, L=3, h=4, ffn=112, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h, ffn) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, targets=None):
        h = self.emb(x) + sinusoidal(x.shape[1], self.emb.embedding_dim, x.device)
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))

class RecurrentCellLM(nn.Module):
    """Elman RNN / GRU / LSTM era baselines: sequential cell, final hidden -> head."""
    def __init__(self, vocab, cell="lstm", d=96, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        if cell == "elman":
            self.cell = nn.RNNCell(d, d)
        elif cell == "gru":
            self.cell = nn.GRUCell(d, d)
        else:
            self.cell = nn.LSTMCell(d, d)
        self.cell_kind = cell
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        d = self.cell.input_size
        h = torch.zeros(B, d, device=x.device)
        c = torch.zeros(B, d, device=x.device) if self.cell_kind == "lstm" else None
        outs = []
        for t in range(T):
            e = self.emb(x[:, t])
            if self.cell_kind == "lstm":
                h, c = self.cell(e, (h, c))
            else:
                h = self.cell(e, h)
            outs.append(h)
        return self.head(self.lnf(torch.stack(outs, dim=1)))

def swa_step_attention(q_x, cache, Wq, Wk, Wv, Wo, h, wsize):
    """One sliding-window self-attention step over a list of cached (k,v) tensors."""
    d = q_x.shape[-1]
    cache.append((Wk(q_x), Wv(q_x)))
    ks = torch.stack([c[0] for c in cache[-wsize:]], dim=1)
    vs = torch.stack([c[1] for c in cache[-wsize:]], dim=1)
    B, D = q_x.shape
    q = Wq(q_x).view(B, h, D // h).unsqueeze(2)          # (B, h, 1, hd)
    k = ks.view(B, -1, h, D // h).transpose(1, 2)         # (B, h, S, hd)
    v = vs.view(B, -1, h, D // h).transpose(1, 2)         # (B, h, S, hd)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(D // h)
    att = att.softmax(-1)
    y = (att @ v).squeeze(2).reshape(B, D)
    return Wo(y)

class TokenMergeLM(nn.Module):
    """Ablation: recurrent decoder merging raw token embedding with state; SWA self-attn only."""
    def __init__(self, vocab, d=56, L=3, h=4, ffn=112, window=8, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.merge = nn.Linear(2 * d, d, bias=False)
        self.h, self.window = h, window
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.wq = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.wk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.wv = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.wo = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.f1 = nn.ModuleList([nn.Linear(d, ffn, bias=False) for _ in range(L)])
        self.f2 = nn.ModuleList([nn.Linear(ffn, d, bias=False) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        s = torch.zeros(B, self.emb.embedding_dim, device=x.device)
        caches = [[] for _ in self.ln1]
        outs = []
        for t in range(T):
            u = self.merge(torch.cat([self.emb(x[:, t]), s], dim=-1))
            h = u
            for i in range(len(self.ln1)):
                q_x = self.ln1[i](h)
                h = h + swa_step_attention(q_x, caches[i], self.wq[i], self.wk[i], self.wv[i], self.wo[i], self.h, self.window)
                h = h + self.f2[i](F.gelu(self.f1[i](self.ln2[i](h))))
            s = h
            outs.append(s)
        return self.head(self.lnf(torch.stack(outs, dim=1)))

class RLTLite(nn.Module):
    """Faithful small-scale RLT: causal encoder -> global KV memory;
    recurrent decoder with per-layer cross-attention to memory + layerwise SWA cache
    + feedback of final hidden state. Full BPTT."""
    def __init__(self, vocab, d=56, LE=2, LD=2, h=4, ffn=112, window=8, **kw):
        super().__init__()
        self.d, self.h, self.window, self.LD = d, h, window, LD
        self.emb = nn.Embedding(vocab, d)
        self.enc_blocks = nn.ModuleList([Block(d, h, ffn) for _ in range(LE)])
        self.enc_ln = nn.LayerNorm(d)
        self.mk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.mv = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.merge = nn.Linear(2 * d, d, bias=False)
        self.ln_cr = nn.ModuleList([nn.LayerNorm(d) for _ in range(LD)])
        self.cq = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.ck = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.cv = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.co = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(LD)])
        self.wq = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.wk = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.wv = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.wo = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(LD)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(LD)])
        self.f1 = nn.ModuleList([nn.Linear(d, ffn, bias=False) for _ in range(LD)])
        self.f2 = nn.ModuleList([nn.Linear(ffn, d, bias=False) for _ in range(LD)])
        self.s0 = nn.Parameter(torch.zeros(d))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape
        e = self.emb(x)
        for b in self.enc_blocks:
            e = b(e)               # causal encoder, parallel over tokens
        mem_kv = self.enc_ln(e)     # (B,T,d) global encoder memory
        # cross-attn K/V are parameter-only transforms of the encoder memory:
        # precompute for ALL positions once (identical math to per-step projection)
        M_k = [self.ck[i](self.mk[i](mem_kv)) for i in range(self.LD)]
        M_v = [self.cv[i](self.mv[i](mem_kv)) for i in range(self.LD)]
        s = self.s0.unsqueeze(0).expand(B, -1)
        caches = [[] for _ in range(self.LD)]
        outs = []
        for t in range(T):
            u = self.merge(torch.cat([mem_kv[:, t], s], dim=-1))
            h = u
            for i in range(self.LD):
                # cross-attention to global encoder memory M_{<=t}
                q_x = self.ln_cr[i](h)
                q = self.cq[i](q_x).view(B, self.h, self.d // self.h).unsqueeze(2)   # (B,h,1,hd)
                k = M_k[i][:, : t + 1].view(B, t + 1, self.h, -1).transpose(1, 2)
                v = M_v[i][:, : t + 1].view(B, t + 1, self.h, -1).transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d // self.h)
                att = att.softmax(-1)
                h = h + self.co[i]((att @ v).squeeze(2).reshape(B, self.d))
                # sliding-window self-attention over decoder activations
                q_x = self.ln1[i](h)
                h = h + swa_step_attention(q_x, caches[i], self.wq[i], self.wk[i], self.wv[i], self.wo[i], self.h, self.window)
                h = h + self.f2[i](F.gelu(self.f1[i](self.ln2[i](h))))
            s = h
            outs.append(s)
        return self.head(self.lnf(torch.stack(outs, dim=1)))


MODELS = {
    "elman": lambda v, cfg: RecurrentCellLM(v, cell="elman", **cfg["elman"]),
    "lstm": lambda v, cfg: RecurrentCellLM(v, cell="lstm", **cfg["lstm"]),
    "gru": lambda v, cfg: RecurrentCellLM(v, cell="gru", **cfg["gru"]),
    "transformer": lambda v, cfg: TransformerLM(v, **cfg["transformer"]),
    "token_merge": lambda v, cfg: TokenMergeLM(v, **cfg["token_merge"]),
    "rlt": lambda v, cfg: RLTLite(v, **cfg["rlt"]),
}

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def evaluate(model, eval_sets, device):
    model.eval()
    accs, times = {}, {}
    with torch.no_grad():
        for ops, (seqs, anss) in eval_sets.items():
            seqs, anss = seqs.to(device), anss.to(device)
            n = seqs.shape[0]
            correct = 0
            t0 = time.perf_counter()
            for i in range(0, n, 256):
                xb, ab = seqs[i:i + 256], anss[i:i + 256]
                pred = model(xb)[:, -1].argmax(-1)
                correct += (pred == ab).sum().item()
            accs[ops] = 100.0 * correct / n
            times[ops] = (time.perf_counter() - t0) / n * 1e3
    model.train()
    return accs, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["parity", "five"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train_ops", type=int, default=32)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(1000 + args.seed)
    device = "cpu"
    torch.set_num_threads(2)

    vocab = {"parity": 5, "five": 11}[args.task]
    BOS = vocab - 2
    cfg = {
        "elman": {"d": 192},
        "lstm": {"d": 96},
        "gru": {"d": 112},
        "transformer": {"d": 56, "L": 3, "h": 4, "ffn": 112},
        "token_merge": {"d": 56, "L": 3, "h": 4, "ffn": 112, "window": 8},
        "rlt": {"d": 40, "LE": 2, "LD": 2, "h": 4, "ffn": 80, "window": 8},
    }
    auto = gen_automaton(rng) if args.task == "five" else None
    model = MODELS[args.model](vocab, cfg)
    nparams = count_params(model)
    print(f"[{args.model}/{args.task}/s{args.seed}] params={nparams}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup, total = 200, args.steps
    def lr_fn(s):
        if s < warmup:
            return (s + 1) / warmup
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min((s - warmup) / max(total - warmup, 1), 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    train_rng = np.random.default_rng(2000 + args.seed)
    eval_rng = np.random.default_rng(3000 + args.seed)
    eval_sets = {}
    for ops in (16, 32, 64, 128):
        eval_sets[ops] = make_fixed_eval(args.task, ops, 2048, eval_rng, auto)

    losses, step_ms = [], []
    model.train()
    t_start = time.perf_counter()
    for step in range(args.steps):
        t0 = time.perf_counter()
        seq, ans = make_batch(args.task, args.train_ops, args.batch, train_rng, auto)
        x = torch.cat([torch.full((seq.shape[0], 1), BOS, dtype=torch.long), seq], dim=1)
        logits = model(x)   # (B, T, vocab)
        if args.task == "parity":
            # dense supervision: predict running parity after every bit
            run = seq.cumsum(dim=1) % 2
            loss = F.cross_entropy(logits[:, 1:].reshape(-1, vocab), run.reshape(-1))
        else:
            # sparse final-state supervision: single answer after the full chain
            loss = F.cross_entropy(logits[:, -1], ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        step_ms.append((time.perf_counter() - t0) * 1e3)
        if step % 500 == 0:
            print(f"  step {step} loss {np.mean(losses[-50:]):.4f}", flush=True)

    accs, eval_times = evaluate(model, eval_sets, device)
    out = {
        "model": args.model, "task": args.task, "seed": args.seed,
        "params": nparams, "steps": args.steps, "batch": args.batch,
        "train_ops": args.train_ops,
        "supervision": "dense_running_parity" if args.task == "parity" else "sparse_final_state",
        "acc": accs, "eval_ms_per_program": eval_times,
        "train_ms_per_step": float(np.mean(step_ms[-500:])),
        "final_loss": float(np.mean(losses[-100:])),
        "train_seconds": time.perf_counter() - t_start,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.model}_{args.task}_seed{args.seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)

if __name__ == "__main__":
    main()
