"""在冻结的 15 条视频质量集上评价 Q1-U 与 Q1-F。

人工标注 JSON 格式： [{"sample_id":"video$_$clip", "words":[
{"word":"...", "start":0.12, "end":0.44}, ...]}]. 词语列表必须
与原转写顺序一致，否则拒绝该记录。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from experiment import ROOT


def normalize(word): return re.sub(r"[^a-z0-9]+", "", str(word).lower())


def measure(reference, predicted):
    if len(reference) != len(predicted): raise ValueError("词语数量不一致")
    matched = []; ious = []; within100 = []; within200 = []
    for truth, estimate in zip(reference, predicted):
        if estimate is None:
            ious.append(0.0); within100.append(0.0); within200.append(0.0)
            continue
        s, e = map(float, estimate)
        ts, te = float(truth["start"]), float(truth["end"])
        if not 0 <= ts < te: raise ValueError("参考时间区间无效")
        errors = [abs(s - ts), abs(e - te)]
        matched.extend(errors)
        union = max(e, te) - min(s, ts)
        ious.append(max(0.0, min(e, te) - max(s, ts)) / union if union > 0 else 0.0)
        within100.append(float(max(errors) <= 0.1)); within200.append(float(max(errors) <= 0.2))
    return {"word_count": len(reference), "matched_word_count": len(matched) // 2,
            "word_coverage": len(matched) / (2 * len(reference)),
            "conditional_boundary_mae_seconds": float(np.mean(matched)) if matched else None,
            "interval_iou_missing_as_zero": float(np.mean(ious)),
            "both_boundaries_within_100ms": float(np.mean(within100)),
            "both_boundaries_within_200ms": float(np.mean(within200))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True, help="人工词级时间标注 JSON 路径")
    ap.add_argument("--q1", type=Path, default=ROOT / "outputs" / "q1", help="Q1 输出目录")
    args = ap.parse_args()
    frozen = json.loads((args.q1 / "frozen_audit_20.json").read_text(encoding="utf-8"))
    wanted = {x["sample_id"] for x in frozen if x["role"] == "blind_quality"}
    refs = {x["sample_id"]: x for x in json.loads(args.annotations.read_text(encoding="utf-8"))}
    if set(refs) & wanted != wanted: raise ValueError(f"缺少人工标注： {sorted(wanted-set(refs))}")
    base = {x["sample_id"]: x for x in map(json.loads, (args.q1 / "q1_u_100.jsonl").open(encoding="utf-8"))}
    forced_path = args.q1 / "q1_extraction_100.jsonl"
    if not forced_path.exists(): raise FileNotFoundError("尚未执行 Q1 强制对齐")
    forced = {x["sample_id"]: x for x in map(json.loads, forced_path.open(encoding="utf-8"))}
    rows = []
    for sample_id in sorted(wanted):
        reference = refs[sample_id]["words"]
        expected = [w["word"] for w in base[sample_id]["q1_u_intervals"]]
        if [normalize(w["word"]) for w in reference] != [normalize(w) for w in expected]:
            raise ValueError(f"标注与转写的词序不一致： {sample_id}")
        uniform = [[w["start"], w["end"]] for w in base[sample_id]["q1_u_intervals"]]
        aligned = forced[sample_id].get("word_intervals", [None] * len(expected))
        for method, predictions in (("Q1-U", uniform), ("Q1-F", aligned)):
            row = measure(reference, predictions)
            row.update({"sample_id": sample_id, "video_id": base[sample_id]["video_id"], "method": method})
            rows.append(row)
    summary = {}
    for method in ("Q1-U", "Q1-F"):
        method_rows = [x for x in rows if x["method"] == method]
        summary[method] = {key: float(np.mean([r[key] for r in method_rows if r[key] is not None]))
                           for key in ("word_coverage", "conditional_boundary_mae_seconds",
                                       "interval_iou_missing_as_zero", "both_boundaries_within_100ms",
                                       "both_boundaries_within_200ms")}
    # 按视频而非按词进行配对重抽样。
    rng = np.random.default_rng(20260923)
    videos = sorted({x["video_id"] for x in rows})
    differences = []
    by = {(v, method): [x for x in rows if x["video_id"] == v and x["method"] == method]
          for v in videos for method in ("Q1-U", "Q1-F")}
    for _ in range(2000):
        sampled = rng.choice(videos, len(videos), replace=True)
        u = [x["interval_iou_missing_as_zero"] for v in sampled for x in by[(v, "Q1-U")]]
        f = [x["interval_iou_missing_as_zero"] for v in sampled for x in by[(v, "Q1-F")]]
        differences.append(float(np.mean(np.asarray(f) - np.asarray(u))))
    summary["paired_interval_iou_gain_95pct_ci"] = np.quantile(differences, [0.025, 0.975]).tolist()
    out = {"n_quality_videos": len(wanted), "note": "Boundary MAE conditional on matched words; missing words count as zero IoU",
           "summary": summary, "per_video": rows}
    path = args.q1 / "quality_evaluation.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "videos": len(wanted)}, ensure_ascii=False))


if __name__ == "__main__": main()
