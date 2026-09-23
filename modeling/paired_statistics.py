"""对三项预先登记的比较执行视频聚类配对自助抽样。

需要 run_protocol.py test 生成的已锁定 E-main 逐样本预测文件。
每次按视频编号重抽样，在每个条件下重新计算三分类 Macro-F1，
然后对条件及随机种子取等权平均。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment import ROOT


PAIRS = (("H2", "F21", "F11"), ("H3", "F21", "F20"), ("H4", "D1", "F21"))


def macro_f1(confusion):
    tp = np.diagonal(confusion, axis1=-2, axis2=-1)
    actual = confusion.sum(axis=-1)
    predicted = confusion.sum(axis=-2)
    denom = actual + predicted
    per_class = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)
    return per_class.mean(axis=-1)


def load_run(path):
    # 只在内存中保存汇总统计量，避免构造 727×84 的长表。
    cells = defaultdict(lambda: [np.zeros((3, 3), dtype=np.int32), 0.0, 0])
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            condition = row["condition_id"]
            if condition.endswith("_clean"): continue
            key = (condition, row["video_id"])
            item = cells[key]
            truth = int(row["true_polarity"]); pred = int(row["pred_polarity"])
            item[0][truth, pred] += 1
            item[1] += abs(float(row["true_intensity"]) - float(row["pred_intensity"]))
            item[2] += 1
    return cells


def arrays(cells, conditions, videos):
    confusion = np.zeros((len(conditions), len(videos), 3, 3), np.int32)
    errors = np.zeros((len(conditions), len(videos)), np.float64)
    counts = np.zeros((len(conditions), len(videos)), np.int32)
    ci = {key: i for i, key in enumerate(conditions)}
    vi = {key: i for i, key in enumerate(videos)}
    for (condition, video), (cm, err, n) in cells.items():
        confusion[ci[condition], vi[video]] = cm
        errors[ci[condition], vi[video]] = err
        counts[ci[condition], vi[video]] = n
    return confusion, errors, counts


def evaluate(weights, stats):
    confusion, errors, counts = stats
    # 权重形状：[自助抽样次数, 视频]；指标形状：[自助抽样次数, 条件]
    cm = np.einsum("bv,cvij->bcij", weights, confusion, optimize=True)
    err = np.einsum("bv,cv->bc", weights, errors, optimize=True)
    n = np.einsum("bv,cv->bc", weights, counts, optimize=True)
    f1 = macro_f1(cm)
    mae = np.divide(err, n, out=np.full_like(err, np.nan), where=n > 0)
    return np.nanmean(f1, axis=1), np.nanmean(mae, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "official", help="已锁定结果目录")
    ap.add_argument("--replicates", type=int, default=2000, help="自助抽样重复次数")
    args = ap.parse_args()
    lock = json.loads((args.out / "protocol_lock.json").read_text(encoding="utf-8"))
    if lock["status"] != "locked_verified": raise ValueError("需要已核验且已锁定的测试结果")
    wanted = {model for _, a, b in PAIRS for model in (a, b)}
    records = defaultdict(dict); all_conditions = set(); all_videos = set()
    for run_id, rec in lock["all_valid_results"].items():
        if rec["model"] not in wanted: continue
        path = args.out / "predictions" / f"{run_id}_test_E-main.csv.gz"
        if not path.exists(): raise FileNotFoundError(path)
        cells = load_run(path)
        records[rec["model"]][int(rec["seed"])] = cells
        all_conditions.update(c for c, _ in cells)
        all_videos.update(v for _, v in cells)
    seeds = (17, 29, 43)
    conditions = sorted(all_conditions); videos = sorted(all_videos)
    if len(conditions) != 84: raise ValueError(f"预期 84 个缺失条件，实际为 {len(conditions)}")
    if not videos: raise ValueError("没有视频编号")
    tensors = {}
    for model in wanted:
        if set(records[model]) != set(seeds): raise ValueError(f"随机种子结果不完整： {model}")
        tensors[model] = {seed: arrays(records[model][seed], conditions, videos) for seed in seeds}
    rng = np.random.default_rng(20260923)
    draws = rng.integers(0, len(videos), size=(args.replicates, len(videos)))
    weights = np.empty((args.replicates, len(videos)), np.float64)
    for i, draw in enumerate(draws): weights[i] = np.bincount(draw, minlength=len(videos))
    actual = np.ones((1, len(videos)), np.float64)
    report = {"cluster": "video_id", "video_count": len(videos), "conditions": len(conditions),
              "seeds": seeds, "bootstrap_replicates": args.replicates, "comparisons": {}}
    for hypothesis, treatment, control in PAIRS:
        observed_f1 = []; observed_mae = []
        boot_f1 = np.zeros(args.replicates); boot_mae = np.zeros(args.replicates)
        for seed in seeds:
            tf1, tmae = evaluate(actual, tensors[treatment][seed]); cf1, cmae = evaluate(actual, tensors[control][seed])
            observed_f1.append(float(tf1[0] - cf1[0])); observed_mae.append(float(tmae[0] - cmae[0]))
            # 分块处理 2000 次重抽样，限制内存占用。
            for lo in range(0, args.replicates, 50):
                w = weights[lo:lo + 50]
                bf1, bmae = evaluate(w, tensors[treatment][seed])
                af1, amae = evaluate(w, tensors[control][seed])
                boot_f1[lo:lo + len(w)] += (bf1 - af1) / len(seeds)
                boot_mae[lo:lo + len(w)] += (bmae - amae) / len(seeds)
        report["comparisons"][hypothesis] = {
            "treatment": treatment, "control": control,
            "macro_f1_difference": float(np.mean(observed_f1)),
            "macro_f1_seed_differences": observed_f1,
            "macro_f1_ci95": np.quantile(boot_f1, [0.025, 0.975]).tolist(),
            "macro_f1_ci98_33_bonferroni": np.quantile(boot_f1, [1 / 120, 119 / 120]).tolist(),
            "mae_difference_treatment_minus_control": float(np.mean(observed_mae)),
            "mae_seed_differences": observed_mae,
            "mae_ci95": np.quantile(boot_mae, [0.025, 0.975]).tolist()}
    path = args.out / "metrics" / "primary_paired_bootstrap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "comparisons": list(report["comparisons"])}, ensure_ascii=False))


if __name__ == "__main__": main()
