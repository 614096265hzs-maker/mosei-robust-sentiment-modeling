"""Aligned MOSEI experiment runner. Run `python experiment.py --help`.

The official route requires a verified frozen BERT cache made by `encode`.
`--source precomputed` is an explicitly provisional route for attachment 2/4.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MSA_DATA_ROOT", ROOT)).expanduser().resolve()
DESIGN = ROOT / "protocol"
DATA = DATA_ROOT / "E题数据"
ALIGNED = DATA / "附件2-数据集特征文件" / "aligned_50.pkl"
A3 = DATA / "附件3-模态缺失特征样本" / "对齐版本"
A4 = DATA / "附件4-可解释专项视频样本与特征文件" / "附件4-可解释专项视频样本与特征文件" / "对齐版本"
KEYS = ("T", "A", "V")
FEATURES = {"T": "text", "A": "audio", "V": "vision"}
DIMS = {"T": 768, "A": 74, "V": 35}
MODELS = {
    "B1": ("T", True, "single", "none"), "B2": ("A", True, "single", "none"),
    "B3": ("V", True, "single", "none"), "B4": ("TAV", False, "concat", "none"),
    "B5": ("TAV", True, "concat", "none"), "F00": ("TAV", True, "uniform", "none"),
    "F01": ("TAV", True, "gate", "none"), "F10": ("TAV", True, "uniform", "iid"),
    "F11": ("TAV", True, "gate", "iid"), "F20": ("TAV", True, "uniform", "block"),
    "F21": ("TAV", True, "gate", "block"), "D1": ("TAV", True, "gate", "block"),
}


def load_pickle(path: Path):
    # Some specialty files were pickled with NumPy 2, while the available
    # CPU PyTorch environment has NumPy 1. A module alias permits read-only load.
    import numpy.core as core
    sys.modules.setdefault("numpy._core", core)
    sys.modules.setdefault("numpy._core.numeric", core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", core.multiarray)
    with path.open("rb") as f:
        return pickle.load(f)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*items) -> int:
    s = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")


def body_mask(bert: np.ndarray) -> np.ndarray:
    attn = np.asarray(bert[:, 1, :], dtype=bool)
    lengths = attn.sum(axis=1)
    idx = np.arange(attn.shape[1])[None, :]
    return attn & (idx > 0) & (idx < (lengths[:, None] - 1))


def normalize_fit(split: dict, text: np.ndarray) -> dict:
    c = body_mask(split["text_bert"])
    out = {}
    for k in KEYS:
        raw = np.asarray(text if k == "T" else split[FEATURES[k]], dtype=np.float32)
        observed = c if k == "T" else (c & np.any(raw != 0, axis=2))
        rows = raw[observed].astype(np.float64)
        if len(rows):
            mean = rows.mean(axis=0)
            std = rows.std(axis=0)
        else:
            mean = np.zeros(DIMS[k]); std = np.ones(DIMS[k])
        dead = (~np.isfinite(std)) | (std < 1e-6)
        std[dead] = 1.0
        out[k + "_mean"] = mean.astype(np.float32)
        out[k + "_std"] = std.astype(np.float32)
        out[k + "_dead"] = dead
        out[k + "_observed_count"] = np.array(len(rows))
    y = np.asarray(split["classification_labels"], dtype=np.int64)
    counts = np.bincount(y, minlength=3)
    prior = counts / counts.sum()
    weights = 1 / np.sqrt(counts.astype(np.float64))
    weights /= weights.mean()
    out["prior"] = prior.astype(np.float32)
    out["reg_mean"] = np.array(float(np.mean(split["regression_labels"])), dtype=np.float32)
    out["class_weights"] = weights.astype(np.float32)
    return out


def load_text_cache(path: Path, split: str, n: int) -> np.ndarray:
    with np.load(path, mmap_mode="r") as z:
        key = f"{split}_text"
        if key not in z:
            raise ValueError(f"Text cache missing {key}: {path}")
        x = np.array(z[key], dtype=np.float32)
    if x.shape != (n, 50, 768):
        raise ValueError(f"Incorrect text shape {x.shape} for {split}")
    return x


class Dataset:
    def __init__(self, split: dict, text: np.ndarray, norm: dict, name: str):
        self.name = name
        self.ids = [str(x) for x in split["id"]]
        self.bert = np.asarray(split["text_bert"])
        self.c = body_mask(self.bert)
        self.x = {}; self.o = {}; self.uncertain = {}
        for k in KEYS:
            raw = np.asarray(text if k == "T" else split[FEATURES[k]], dtype=np.float32)
            if raw.shape != (len(self.ids), 50, DIMS[k]):
                raise ValueError(f"{name} {k} shape {raw.shape}")
            avail = self.c if k == "T" else self.c & np.any(raw != 0, axis=2)
            self.uncertain[k] = self.c & ~avail if k != "T" else np.zeros_like(self.c)
            self.o[k] = avail
            z = (raw - norm[k + "_mean"]) / norm[k + "_std"]
            z[~avail] = 0
            z[..., norm[k + "_dead"]] = 0
            self.x[k] = z.astype(np.float32)
        self.y_cls = np.asarray(split.get("classification_labels", np.full(len(self.ids), -1)), dtype=np.int64)
        self.y_reg = np.asarray(split.get("regression_labels", np.full(len(self.ids), np.nan)), dtype=np.float32)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f"Duplicate IDs in {name}")

    def __len__(self): return len(self.ids)

    def batch(self, indices, condition=None, train_key=None, aug="none", device="cpu"):
        ids = [self.ids[i] for i in indices]
        masks = {k: self.o[k][indices].copy() for k in KEYS}
        removed = {k: np.zeros_like(masks[k]) for k in KEYS}
        meta = []
        for j, idx in enumerate(indices):
            if condition is not None:
                spec = condition
                rng = np.random.default_rng(stable_seed(spec.get("mask_seed"), self.name, self.ids[idx], spec["condition_id"]))
                style = "block"
            elif train_key is not None and aug != "none":
                rng = np.random.default_rng(stable_seed(*train_key, self.ids[idx]))
                if rng.random() < 0.3:
                    spec = None
                else:
                    spec = {"modalities": rng.choice(["T", "A", "V", "TA", "TV", "AV", "TAV"]),
                            "rate_nominal": float(rng.choice([0.1, 0.3, 0.5, 0.7])),
                            "position": rng.choice(["start", "middle", "end", "random"])}
                style = aug
            else:
                spec = None; rng = None; style = "none"
            info = {"sample_id": self.ids[idx], "body_length": int(self.c[idx].sum()),
                    "nominal_rate": 0.0, "actual_removed": {}, "actual_rate": {},
                    "removed_indices": {}, "observed_count": {}, "window": None}
            if spec and spec.get("modalities"):
                coords = np.flatnonzero(self.c[idx])
                L = len(coords)
                if L:
                    rate = float(spec["rate_nominal"])
                    width = min(L, max(1, math.ceil(rate * L)))
                    pos = spec["position"]
                    start = 0 if pos == "start" else (L - width) // 2 if pos == "middle" else L - width if pos == "end" else int(rng.integers(0, L - width + 1))
                    window = coords[start:start + width]
                    info["window"] = [int(start), int(start + width)]
                    info["nominal_rate"] = width / L
                    for k in spec["modalities"]:
                        take = window[self.o[k][idx, window]]
                        if style == "iid" and len(take):
                            take = rng.choice(np.flatnonzero(self.o[k][idx]), size=len(take), replace=False)
                        masks[k][j, take] = False
                        removed[k][j, take] = True
            for k in KEYS:
                count = int(removed[k][j].sum())
                denom = int(self.o[k][idx].sum())
                info["actual_removed"][k] = count
                info["actual_rate"][k] = count / denom if denom else None
                info["removed_indices"][k] = np.flatnonzero(removed[k][j]).tolist()
                info["observed_count"][k] = denom
            meta.append(info)
        xs = {k: torch.as_tensor(self.x[k][indices].copy(), device=device) for k in KEYS}
        ms = {k: torch.as_tensor(masks[k], device=device) for k in KEYS}
        for k in KEYS: xs[k].masked_fill_(~ms[k].unsqueeze(-1), 0)
        c = torch.as_tensor(self.c[indices], device=device)
        ys = (torch.as_tensor(self.y_cls[indices], device=device), torch.as_tensor(self.y_reg[indices], device=device))
        return xs, ms, c, ys, meta


class Predictor(nn.Module):
    def __init__(self, model_id: str, hidden: int, dropout: float, prior, reg_mean):
        super().__init__()
        modalities, temporal, fusion, _ = MODELS[model_id]
        self.model_id = model_id; self.modalities = modalities; self.temporal = temporal; self.fusion = fusion
        self.project = nn.ModuleDict({k: nn.Linear(DIMS[k], hidden) for k in modalities})
        self.enc = nn.ModuleDict()
        if temporal:
            for k in modalities:
                layer = nn.TransformerEncoderLayer(hidden, 4, 4 * hidden, dropout, activation="gelu", batch_first=True)
                self.enc[k] = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.gate = nn.Sequential(nn.Linear(hidden + 3, 32), nn.GELU(), nn.Linear(32, 1)) if fusion == "gate" else None
        self.combine = nn.Linear(3 * hidden, hidden) if fusion == "concat" else None
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(64, 3); self.reg = nn.Linear(64, 1)
        self.register_buffer("prior_log", torch.log(torch.as_tensor(prior, dtype=torch.float32).clamp_min(1e-8)))
        self.register_buffer("prior_reg", torch.as_tensor(float(reg_mean), dtype=torch.float32))
        pe = torch.zeros(50, hidden)
        pos = torch.arange(50).unsqueeze(1); div = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000) / hidden))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("position", pe)

    @staticmethod
    def longest_gap(mask, content):
        vals = []
        for m, c in zip(mask.detach().cpu().numpy(), content.detach().cpu().numpy()):
            run = best = 0
            for flag in (~m)[c]:
                run = run + 1 if flag else 0; best = max(best, run)
            vals.append(best / max(int(c.sum()), 1))
        return torch.as_tensor(vals, device=mask.device, dtype=torch.float32)

    def forward(self, xs, masks, content):
        vectors = {}; flags = []; gate_logits = []
        for k in self.modalities:
            mask = masks[k].bool(); available = mask.any(dim=1)
            h = self.project[k](xs[k])
            h = h.masked_fill(~mask.unsqueeze(-1), 0)
            if self.temporal:
                safe_mask = mask.clone(); safe_mask[~available, 0] = True
                h = self.enc[k](h + self.position.unsqueeze(0) * safe_mask.unsqueeze(-1), src_key_padding_mask=~safe_mask)
            h = h.masked_fill(~mask.unsqueeze(-1), 0)
            z = h.sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            vectors[k] = z
            flags.append(available)
            if self.gate is not None:
                coverage = mask.sum(dim=1) / content.sum(dim=1).clamp_min(1)
                longest = self.longest_gap(mask, content)
                q = torch.stack((coverage, longest, available.float()), dim=1)
                gate_logits.append(self.gate(torch.cat((z, q), dim=1)).squeeze(1))
        flags = torch.stack(flags, dim=1)
        if self.fusion == "concat":
            padded = [vectors[k] if k in vectors else torch.zeros_like(next(iter(vectors.values()))) for k in KEYS]
            fused = self.combine(torch.cat(padded, dim=1))
            weights = flags.float() / flags.float().sum(dim=1, keepdim=True).clamp_min(1)
        else:
            stack = torch.stack([vectors[k] for k in self.modalities], dim=1)
            if self.gate is None: weights = flags.float() / flags.float().sum(dim=1, keepdim=True).clamp_min(1)
            else:
                logits = torch.stack(gate_logits, dim=1).masked_fill(~flags, -1e9)
                weights = F.softmax(logits, dim=1) * flags
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
            fused = (stack * weights.unsqueeze(-1)).sum(dim=1)
        h = self.head(fused)
        cls = self.cls(h); reg = 3 * torch.tanh(self.reg(h)).squeeze(1)
        empty = ~flags.any(dim=1)
        cls = torch.where(empty[:, None], self.prior_log[None, :], cls)
        reg = torch.where(empty, self.prior_reg, reg)
        return cls, reg, weights, empty


def metric(yc, yr, prob, pred):
    yc = np.asarray(yc, dtype=int); yr = np.asarray(yr, dtype=float)
    cls = np.argmax(prob, axis=1)
    p, r, f, support = precision_recall_fscore_support(yc, cls, labels=[0, 1, 2], zero_division=0)
    corr = None if np.ptp(yr) < 1e-7 or np.ptp(pred) < 1e-7 else float(np.corrcoef(yr, pred)[0, 1])
    return {"n": len(yc), "accuracy": float(accuracy_score(yc, cls)),
            "macro_f1": float(f1_score(yc, cls, labels=[0, 1, 2], average="macro", zero_division=0)),
            "mae": float(np.mean(np.abs(yr - pred))), "pearson_r": corr,
            "per_class": [{"label": i, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i in range(3)],
            "confusion": confusion_matrix(yc, cls, labels=[0, 1, 2]).tolist()}


def selection_score(metrics):
    scores = [0.5 * m["macro_f1"] + 0.5 * (1 - m["mae"] / 6) for m in metrics]
    return 0.5 * scores[0] + 0.5 * float(np.mean(scores[1:]))


def read_conditions(suite):
    with (DESIGN / "评估条件清单.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if json.loads(line)["suite"] == suite]


@torch.no_grad()
def predict(model, data, condition=None, batch_size=64, device="cpu", rows=False):
    model.eval(); probs = []; regs = []; fallbacks = []; all_rows = []
    for lo in range(0, len(data), batch_size):
        ix = np.arange(lo, min(lo + batch_size, len(data)))
        xs, ms, c, _, meta = data.batch(ix, condition=condition, device=device)
        cls, reg, weights, fallback = model(xs, ms, c)
        pp = cls.softmax(dim=1).cpu().numpy(); rr = reg.cpu().numpy()
        probs.append(pp); regs.append(rr); fallbacks.append(fallback.cpu().numpy())
        if rows:
            ww = weights.cpu().numpy()
            for j, i in enumerate(ix):
                all_rows.append({"sample_id": data.ids[i], "p_negative": float(pp[j, 0]), "p_neutral": float(pp[j, 1]), "p_positive": float(pp[j, 2]),
                                 "pred_polarity": int(np.argmax(pp[j])), "pred_intensity": float(rr[j]), "fallback": bool(fallback[j]),
                                 "weights": {k: float(ww[j, h]) for h, k in enumerate(model.modalities)}, "mask": meta[j]})
    return np.concatenate(probs), np.concatenate(regs), np.concatenate(fallbacks), all_rows


def data_bundle(source, cache, splits=("train", "valid")):
    raw = load_pickle(ALIGNED)
    if source == "verified":
        if not cache: raise ValueError("Verified mode requires --text-cache from encode")
        meta = json.loads(cache.with_suffix(".json").read_text(encoding="utf-8"))
        if not meta.get("verified") or meta.get("data_sha256") != sha_file(ALIGNED):
            raise ValueError("Text cache verification or data hash failed")
        texts = {s: load_text_cache(cache, s, len(raw[s]["id"])) for s in splits}
    else:
        texts = {s: raw[s]["text"] for s in splits}
    norm = normalize_fit(raw["train"], texts["train"])
    datasets = {s: Dataset(raw[s], texts[s], norm, s) for s in splits}
    return datasets, norm


def save_norm(out, norm, source, cache):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "normalization.npz"
    np.savez(path, **norm)
    (out / "data_source.json").write_text(json.dumps({"source": source, "text_cache": str(cache) if cache else None,
        "aligned_sha256": sha_file(ALIGNED), "protocol_sha256": sha_file(DESIGN / "实验协议.json")}, indent=2), encoding="utf-8")


def train_one(args, datasets, norm, model_id, seed, hidden, lr, dropout, run_name, teacher_path=None):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(args.threads)
    device = args.device
    model = Predictor(model_id, hidden, dropout, norm["prior"], norm["reg_mean"]).to(device)
    teacher = None
    if model_id == "D1":
        if not teacher_path or not teacher_path.exists(): raise ValueError("D1 requires same-seed F01 checkpoint")
        cp = torch.load(teacher_path, map_location=device, weights_only=False)
        teacher = Predictor("F01", hidden, dropout, norm["prior"], norm["reg_mean"]).to(device)
        teacher.load_state_dict(cp["state"]); teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    weights = torch.as_tensor(norm["class_weights"], device=device)
    train = datasets["train"]; valid = datasets["valid"]
    conditions = read_conditions("V-select")
    best = -float("inf"); stale = 0; history = []; t0 = time.time()
    path = args.out / "checkpoints" / f"{run_name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); rng = np.random.default_rng(stable_seed(seed, epoch, "order")); order = rng.permutation(len(train))
        losses = []
        for lo in range(0, len(order), args.batch_size):
            ix = order[lo:lo + args.batch_size]
            xs, masks, c, (yc, yr), meta = train.batch(ix, train_key=(seed, epoch), aug=MODELS[model_id][3], device=device)
            cls, reg, _, _ = model(xs, masks, c)
            loss = F.cross_entropy(cls, yc, weight=weights) + F.huber_loss(reg, yr, delta=1)
            if teacher is not None:
                changed = torch.as_tensor([sum(m["actual_removed"].values()) > 0 for m in meta], device=device)
                if changed.any():
                    full_x, full_m, full_c, _, _ = train.batch(ix, device=device)
                    with torch.no_grad(): tc, tr, _, _ = teacher(full_x, full_m, full_c)
                    kd = F.kl_div(F.log_softmax(cls[changed], dim=1), F.softmax(tc[changed], dim=1), reduction="batchmean")
                    kd += torch.mean(torch.abs(reg[changed] - tr[changed])) / 6
                    loss = loss + 0.2 * kd
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step()
            losses.append(float(loss.detach()))
        evals = []
        for cond in conditions:
            p, r, _, _ = predict(model, valid, cond, args.batch_size, device)
            evals.append(metric(valid.y_cls, valid.y_reg, p, r))
        score = selection_score(evals)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "selection_score": score,
                        "clean_macro_f1": evals[0]["macro_f1"], "clean_mae": evals[0]["mae"]})
        print(f"{run_name} epoch={epoch+1} loss={history[-1]['loss']:.4f} S={score:.4f}", flush=True)
        if score >= best + 1e-4:
            best = score; stale = 0
            torch.save({"state": model.state_dict(), "model_id": model_id, "hidden": hidden, "dropout": dropout,
                        "seed": seed, "best_epoch": epoch + 1, "selection_score": score, "source": args.source}, path)
        else: stale += 1
        if epoch + 1 >= max(10, args.min_epochs) and stale >= 8: break
    result = {"run_id": run_name, "model": model_id, "seed": seed, "hidden_dim": hidden, "learning_rate": lr,
              "dropout": dropout, "source": args.source, "checkpoint": str(path), "history": history,
              "best_score": best, "seconds": time.time() - t0,
              "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    p = args.out / "runs" / f"{run_name}.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_model(path, norm, device):
    cp = torch.load(path, map_location=device, weights_only=False)
    model = Predictor(cp["model_id"], cp["hidden"], cp["dropout"], norm["prior"], norm["reg_mean"]).to(device)
    model.load_state_dict(cp["state"]); model.eval()
    return model, cp


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        fields = ["sample_id", "p_negative", "p_neutral", "p_positive", "pred_polarity", "pred_intensity", "fallback", "weights", "mask"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows:
            row = row.copy()
            for k in ("weights", "mask"): row[k] = json.dumps(row[k], ensure_ascii=False)
            w.writerow(row)


def run_evaluate(args):
    if args.split == "test":
        lock_path = args.out / "protocol_lock.json"
        if not lock_path.exists(): raise ValueError("Test evaluation requires protocol_lock.json")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "locked_verified" or args.source != "verified":
            raise ValueError("Test evaluation requires a verified locked protocol")
        if sha_file(ALIGNED) != lock["data_sha256"]:
            raise ValueError("Aligned data changed after protocol lock")
        if sha_file(DESIGN / "评估条件清单.jsonl") != lock["condition_sha256"]:
            raise ValueError("Evaluation conditions changed after protocol lock")
        if args.text_cache is None or sha_file(args.text_cache) != lock["text_cache_sha256"]:
            raise ValueError("Text cache changed after protocol lock")
        if (args.checkpoint.name not in lock["checkpoint_sha256"] or
                sha_file(args.checkpoint) != lock["checkpoint_sha256"][args.checkpoint.name]):
            raise ValueError("Checkpoint is not locked")
    datasets, norm = data_bundle(args.source, args.text_cache, ("train", args.split))
    model, cp = load_model(args.checkpoint, norm, args.device)
    if cp["source"] != args.source: raise ValueError("Checkpoint/text source mismatch")
    data = datasets[args.split]; conditions = read_conditions(args.suite)
    summary = []
    for cond in conditions:
        p, r, fb, rows = predict(model, data, cond, args.batch_size, args.device, args.save_predictions)
        m = metric(data.y_cls, data.y_reg, p, r); m.update({"condition_id": cond["condition_id"], "fallback_rate": float(fb.mean())})
        summary.append(m)
        if args.save_predictions:
            write_rows(args.out / "predictions" / f"{args.checkpoint.stem}_{args.split}_{cond['condition_id']}.csv.gz", rows)
    path = args.out / "metrics" / f"{args.checkpoint.stem}_{args.split}_{args.suite}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "conditions": len(summary), "clean": summary[0]}, ensure_ascii=False))


def run_b0(args):
    datasets, norm = data_bundle(args.source, args.text_cache, ("train", "valid"))
    data = datasets["valid"]
    p = np.broadcast_to(norm["prior"], (len(data), 3))
    r = np.full(len(data), norm["reg_mean"])
    out = {"model": "B0", "split": "valid", "metrics": metric(data.y_cls, data.y_reg, p, r),
           "prior": norm["prior"].tolist(), "reg_mean": float(norm["reg_mean"]), "source": args.source}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "b0_valid.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    save_norm(args.out, norm, args.source, args.text_cache)
    print(json.dumps(out, ensure_ascii=False))


def run_q0(args):
    torch.manual_seed(7); rng = np.random.default_rng(7)
    prior = np.array([0.3, 0.2, 0.5], np.float32)
    for model_id in ("B1", "B2", "B3", "B4", "F00", "F01", "F21"):
        model = Predictor(model_id, 64, 0.1, prior, 0.25).eval()
        xs = {k: torch.from_numpy(rng.normal(size=(3, 50, DIMS[k])).astype(np.float32)) for k in KEYS}
        masks = {k: torch.zeros(3, 50, dtype=torch.bool) for k in KEYS}
        content = torch.zeros(3, 50, dtype=torch.bool)
        content[:, 1:5] = True
        for k in KEYS: masks[k][0, 1:5] = True; masks[k][1, 2] = True
        for k in KEYS: xs[k].masked_fill_(~masks[k].unsqueeze(-1), 0)
        with torch.no_grad(): a = model(xs, masks, content)
        altered = {k: v.clone() for k, v in xs.items()}
        for k in KEYS: altered[k][~masks[k]] = 1000
        with torch.no_grad(): b = model(altered, masks, content)
        assert torch.max(torch.abs(a[0] - b[0])) < 1e-5
        assert torch.max(torch.abs(a[1] - b[1])) < 1e-5
        assert torch.isfinite(a[0]).all() and torch.isfinite(a[1]).all()
        assert torch.allclose(a[0][-1].softmax(0), torch.from_numpy(prior), atol=1e-6)
        assert abs(float(a[1][-1]) - 0.25) < 1e-6
    print("Q0 synthetic masking, fallback, finite-output checks passed")


def run_encode(args):
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise RuntimeError("Install transformers in the selected Python environment") from e
    if not args.encoder or not args.encoder.is_dir(): raise ValueError("--encoder must be a local model directory")
    if args.text_cache.suffix.lower() != ".npz": raise ValueError("--text-cache must end in .npz")
    model = AutoModel.from_pretrained(str(args.encoder), local_files_only=True, trust_remote_code=False).eval().to(args.device)
    if getattr(model.config, "hidden_size", None) != 768: raise ValueError("Encoder hidden size must be 768")
    raw = load_pickle(ALIGNED); arrays = {}
    def encode_bert(bert):
        if not np.all(bert == np.round(bert)): raise ValueError("Noninteger token IDs")
        parts = []
        for lo in range(0, len(bert), args.batch_size):
            b = torch.as_tensor(bert[lo:lo + args.batch_size], device=args.device, dtype=torch.long)
            x = model(input_ids=b[:, 0], attention_mask=b[:, 1], token_type_ids=b[:, 2]).last_hidden_state
            parts.append(x.cpu().numpy().astype(np.float32))
        return np.concatenate(parts)
    with torch.no_grad():
        for name in ("train", "valid", "test"):
            bert = np.asarray(raw[name]["text_bert"])
            arrays[f"{name}_text"] = encode_bert(bert)
            ref = np.asarray(raw[name]["text"], dtype=np.float32)
            active = np.asarray(bert[:, 1, :], dtype=bool)
            mae = float(np.mean(np.abs(arrays[f"{name}_text"][active] - ref[active])))
            print(f"{name}: encoder/reference active-position MAE={mae:.6g}", flush=True)
            if mae > args.max_mae:
                raise ValueError(f"Encoder mismatch on {name}: MAE {mae} > {args.max_mae}")
        for i in range(1, 31):
            name = f"{i:02d}"
            obj = load_pickle(A3 / f"附件3_{name}.pkl")["test"]
            arrays[f"a3_{name}_text"] = encode_bert(np.asarray(obj["text_bert"]))
        for i in range(1, 21):
            name = f"{i:02d}"
            obj = load_pickle(A4 / f"{name}.pkl")
            generated = encode_bert(np.asarray(obj["text_bert"])[None, ...])
            ref = np.asarray(obj["text"], dtype=np.float32)[None, ...]
            active = np.asarray(obj["text_bert"])[1].astype(bool)[None, ...]
            mae = float(np.mean(np.abs(generated[active] - ref[active])))
            if mae > args.max_mae: raise ValueError(f"Encoder mismatch on attachment 4 {name}: {mae}")
            arrays[f"a4_{name}_text"] = generated
    args.text_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.text_cache, **arrays)
    files = sorted(p for p in args.encoder.rglob("*") if p.is_file())
    meta = {"verified": True, "encoder_directory": str(args.encoder), "encoder_files_sha256": {str(p.relative_to(args.encoder)): sha_file(p) for p in files},
            "data_sha256": sha_file(ALIGNED), "max_mae": args.max_mae, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    args.text_cache.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def special_dataset(kind: str, source: str, cache: Path | None, norm: dict) -> Dataset:
    if kind not in ("a3", "a4"): raise ValueError(kind)
    if kind == "a3" and source != "verified":
        raise ValueError("Attachment 3 has no precomputed text. A verified encoder/cache is required")
    parts = {"id": [], "text_bert": [], "audio": [], "vision": [], "text": []}
    z = np.load(cache, mmap_mode="r") if source == "verified" else None
    try:
        for i in range(1, 31 if kind == "a3" else 21):
            name = f"{i:02d}"
            obj = load_pickle(A3 / f"附件3_{name}.pkl")["test"] if kind == "a3" else load_pickle(A4 / f"{name}.pkl")
            parts["id"].append(name)
            for key in ("text_bert", "audio", "vision"):
                val = np.asarray(obj[key])
                parts[key].append(val[0] if kind == "a3" else val)
            parts["text"].append(np.asarray(z[f"{kind}_{name}_text"][0] if z is not None else obj["text"]))
    finally:
        if z is not None: z.close()
    for key in ("text_bert", "audio", "vision", "text"):
        parts[key] = np.stack(parts[key])
    return Dataset(parts, parts["text"], norm, kind)


def run_special(args):
    with np.load(args.out / "normalization.npz") as z: norm = {k: z[k] for k in z.files}
    source_info = json.loads((args.out / "data_source.json").read_text(encoding="utf-8"))
    if args.source != source_info["source"]: raise ValueError("Normalization/source mismatch")
    if args.source == "verified":
        if args.text_cache is None or not args.text_cache.exists(): raise ValueError("Verified text cache missing")
        if Path(source_info["text_cache"]).resolve() != args.text_cache.resolve():
            raise ValueError("Special text cache differs from training cache")
        meta = json.loads(args.text_cache.with_suffix(".json").read_text(encoding="utf-8"))
        if not meta.get("verified") or meta.get("data_sha256") != source_info["aligned_sha256"]:
            raise ValueError("Special text cache verification failed")
    model, cp = load_model(args.checkpoint, norm, args.device)
    if cp["source"] != args.source: raise ValueError("Checkpoint/source mismatch")
    if args.kind == "a3" and args.source != "verified": raise ValueError("Attachment 3 requires verified text encoder")
    data = special_dataset(args.kind, args.source, args.text_cache, norm)
    _, _, _, rows = predict(model, data, batch_size=args.batch_size, device=args.device, rows=True)
    path = args.out / "special" / f"{args.kind}_{args.checkpoint.stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "pred_polarity", "pred_intensity", "p_negative", "p_neutral", "p_positive", "fallback", "weights"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(row[k], ensure_ascii=False) if k == "weights" else row[k] for k in fields})
    assert len(rows) == (30 if args.kind == "a3" else 20)
    assert [r["sample_id"] for r in rows] == [f"{i:02d}" for i in range(1, len(rows) + 1)]
    print(json.dumps({"path": str(path), "rows": len(rows), "source": args.source}, ensure_ascii=False))


def parse():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("q0", "b0", "train", "evaluate", "encode", "special"):
        p = sub.add_parser(name)
        if name in ("b0", "train", "evaluate", "special"):
            p.add_argument("--source", choices=["verified", "precomputed"], default="verified")
            p.add_argument("--text-cache", type=Path)
            p.add_argument("--out", type=Path, default=ROOT / "outputs" / "official")
        if name in ("train", "evaluate", "encode", "special"):
            p.add_argument("--device", default="cpu")
            p.add_argument("--batch-size", type=int, default=32)
        if name == "train":
            p.add_argument("--model", choices=list(MODELS), required=True); p.add_argument("--seed", type=int, default=17)
            p.add_argument("--hidden", type=int, choices=[64, 128], default=64); p.add_argument("--lr", type=float, default=3e-4)
            p.add_argument("--dropout", type=float, choices=[0.1, 0.3], default=0.1)
            p.add_argument("--epochs", type=int, default=50); p.add_argument("--min-epochs", type=int, default=10)
            p.add_argument("--threads", type=int, default=4); p.add_argument("--run-name")
            p.add_argument("--teacher", type=Path)
        if name == "evaluate":
            p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--split", choices=["valid", "test"], default="valid")
            p.add_argument("--suite", choices=["V-select", "E-main", "E-random"], default="V-select")
            p.add_argument("--save-predictions", action="store_true")
        if name == "encode":
            p.add_argument("--encoder", type=Path, required=True); p.add_argument("--text-cache", type=Path, required=True)
            p.add_argument("--max-mae", type=float, default=1e-3)
        if name == "special":
            p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--kind", choices=["a3", "a4"], required=True)
    return ap.parse_args()


def main():
    args = parse()
    if args.command == "q0": run_q0(args)
    elif args.command == "b0": run_b0(args)
    elif args.command == "encode": run_encode(args)
    elif args.command == "evaluate": run_evaluate(args)
    elif args.command == "special": run_special(args)
    elif args.command == "train":
        datasets, norm = data_bundle(args.source, args.text_cache)
        save_norm(args.out, norm, args.source, args.text_cache)
        name = args.run_name or f"{args.model}_s{args.seed}_d{args.hidden}_lr{args.lr}_do{args.dropout}"
        out = train_one(args, datasets, norm, args.model, args.seed, args.hidden, args.lr, args.dropout, name, args.teacher)
        print(json.dumps({k: v for k, v in out.items() if k != "history"}, ensure_ascii=False))


if __name__ == "__main__": main()
