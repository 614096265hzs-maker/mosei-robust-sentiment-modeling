"""对已锁定检查点生成特征位置遮挡解释。

仅输出模型位置索引证据；没有经过核验的
特征到视频映射时，不编造秒级时间。这是在预计算特征上的 P-F 干预。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiment import ROOT, Dataset, data_bundle, load_model, special_dataset, stable_seed


def distance(p, y, q, z):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    jsd = 0.5 * (np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(m, 1e-12))) +
                 np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(m, 1e-12))))
    return float(0.5 * abs(y - z) / 6 + 0.5 * jsd / math.log(2))


@torch.no_grad()
def infer(model, xs, ms, content):
    logits, reg, weights, fallback = model(xs, ms, content)
    return logits.softmax(dim=1).cpu().numpy(), reg.cpu().numpy(), weights.cpu().numpy(), fallback.cpu().numpy()


@torch.no_grad()
def perturb_batch(model, xs, masks, content, edits):
    n = len(edits)
    xx = {k: v.repeat(n, 1, 1) for k, v in xs.items()}
    mm = {k: v.repeat(n, 1) for k, v in masks.items()}
    cc = content.repeat(n, 1)
    for j, changes in enumerate(edits):
        for k, positions in changes.items():
            mm[k][j, positions] = False
            xx[k][j, positions] = 0
    return infer(model, xx, mm, cc)


def contiguous_segments(positions, limit=3):
    positions = sorted(set(int(p) for p in positions))
    if not positions: return []
    result = []; start = prev = positions[0]
    for p in positions[1:]:
        if p != prev + 1:
            result.append([start, prev + 1]); start = p
        prev = p
    result.append([start, prev + 1])
    return result[:limit]


def explain_one(model, data: Dataset, idx: int, window: int, random_repeats: int, device: str):
    xs, masks, content, _, _ = data.batch([idx], device=device)
    p, y, weights, fallback = infer(model, xs, masks, content)
    p0 = p[0]; y0 = float(y[0]); target = int(np.argmax(p0))
    observed = {k: np.flatnonzero(masks[k][0].cpu().numpy()) for k in model.modalities}
    available = {k: len(observed[k]) > 0 for k in model.modalities}
    modal_edits = [{k: observed[k]} for k in model.modalities]
    mp, my, _, _ = perturb_batch(model, xs, masks, content, modal_edits)
    mod = {}
    for j, k in enumerate(model.modalities):
        d = distance(p0, y0, mp[j], float(my[j])) if available[k] else 0.0
        mod[k] = {"distance": d, "signed_intensity_support": y0 - float(my[j]),
                  "signed_class_support": float(p0[target] - mp[j, target]),
                  "available": available[k], "gate_weight": float(weights[0, j])}
    total = sum(v["distance"] for v in mod.values())
    for k in mod: mod[k]["relative_sensitivity"] = mod[k]["distance"] / total if total > 1e-8 and available[k] else None
    edits = []; specs = []
    coords = np.flatnonzero(content[0].cpu().numpy())
    w = min(window, len(coords))
    if w:
        seen = set()
        for k in model.modalities:
            if not available[k]: continue
            for start in range(len(coords) - w + 1):
                pos = tuple(int(v) for v in coords[start:start + w] if v in observed[k])
                if not pos or (k, pos) in seen: continue
                seen.add((k, pos)); edits.append({k: pos}); specs.append((k, pos))
    scores = {k: {int(i): [] for i in observed[k]} for k in model.modalities}
    support = {k: {int(i): [] for i in observed[k]} for k in model.modalities}
    for start in range(0, len(edits), 128):
        pp, yy, _, _ = perturb_batch(model, xs, masks, content, edits[start:start + 128])
        for j in range(len(pp)):
            k, pos = specs[start + j]
            d = distance(p0, y0, pp[j], float(yy[j]))
            signed = float(p0[target] - pp[j, target])
            for i in pos: scores[k][i].append(d); support[k][i].append(signed)
    per_position = {k: {i: float(np.mean(vals)) for i, vals in scores[k].items() if vals} for k in model.modalities}
    per_support = {k: {i: float(np.mean(vals)) for i, vals in support[k].items() if vals} for k in model.modalities}
    ranked = sorted(((score, k, i) for k in model.modalities for i, score in per_position[k].items()), reverse=True)
    budget = max(1, math.ceil(0.2 * len(ranked))) if ranked else 0
    selected = ranked[:budget]
    by_mod = {k: [i for _, kk, i in selected if kk == k] for k in model.modalities}
    main_mod = max((k for k in model.modalities if available[k]), key=lambda k: mod[k]["distance"], default=None)
    top_segments = {k: contiguous_segments(by_mod[k]) for k in model.modalities}
    # 删除预算为 20%；随机对照保持各模态删除位置数一致。
    del_edit = {k: by_mod[k] for k in model.modalities}
    pd, yd, _, _ = perturb_batch(model, xs, masks, content, [del_edit])
    deletion = distance(p0, y0, pd[0], float(yd[0]))
    rng = np.random.default_rng(stable_seed(20260923, data.name, data.ids[idx], "explain_random"))
    random_values = []
    for _ in range(random_repeats):
        edit = {k: rng.choice(observed[k], size=len(by_mod[k]), replace=False).tolist() if by_mod[k] else [] for k in model.modalities}
        rp, ry, _, _ = perturb_batch(model, xs, masks, content, [edit])
        random_values.append(distance(p0, y0, rp[0], float(ry[0])))
    return {"sample_id": data.ids[idx], "pred_polarity": target, "pred_intensity": y0,
            "probabilities": [float(v) for v in p0], "fallback": bool(fallback[0]),
            "model_weights": {k: float(weights[0, j]) for j, k in enumerate(model.modalities)},
            "modality_sensitivity": mod, "main_modality": main_mod,
            "local_window": window, "per_position_distance": per_position, "per_position_signed_class_support": per_support,
            "top20_percent_segments_exclusive_end": top_segments,
            "actual_selected_positions": budget, "deletion_distance_20pct": deletion,
            "random_deletion_distance_20pct_mean": float(np.mean(random_values)),
            "random_control": "same modality counts, independent positions",
            "mapping_status": {k: "unavailable" for k in model.modalities},
            "time_seconds": None, "scope": "precomputed feature positions"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="待解释的模型检查点")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "official", help="输出目录")
    ap.add_argument("--source", choices=["verified", "precomputed"], default="verified",
                    help="文本来源")
    ap.add_argument("--text-cache", type=Path, help="已核验的文本特征缓存路径")
    ap.add_argument("--split", choices=["valid", "test", "a4"], default="a4", help="解释的数据划分")
    ap.add_argument("--window", type=int, choices=[1, 3, 5], default=3, help="遮挡窗口长度")
    ap.add_argument("--random-repeats", type=int, default=20, help="随机对照重复次数")
    ap.add_argument("--device", default="cpu", help="计算设备")
    ap.add_argument("--limit", type=int, help="最多解释的样本数")
    args = ap.parse_args()
    with np.load(args.out / "normalization.npz") as z: norm = {k: z[k] for k in z.files}
    model, cp = load_model(args.checkpoint, norm, args.device)
    if cp["source"] != args.source: raise ValueError("检查点与数据来源不一致")
    if args.split == "a4": data = special_dataset("a4", args.source, args.text_cache, norm)
    else: data = data_bundle(args.source, args.text_cache, ("train", args.split))[0][args.split]
    if args.split in ("valid", "test"):
        rng = np.random.default_rng(20260923)
        selected = []
        for label in range(3):
            group = np.flatnonzero(data.y_cls == label)
            selected.extend(rng.choice(group, size=min(40, len(group)), replace=False).tolist())
        indices = sorted(selected)
    else: indices = list(range(len(data)))
    if args.limit is not None: indices = indices[:args.limit]
    results = []
    for i in indices:
        results.append(explain_one(model, data, i, args.window, args.random_repeats, args.device))
        print(f"已解释 {len(results)}/{len(indices)} {data.ids[i]}", flush=True)
    path = args.out / "explanations" / f"{args.checkpoint.stem}_{args.split}_w{args.window}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"path": str(path), "rows": len(results)}, ensure_ascii=False))


if __name__ == "__main__": main()
