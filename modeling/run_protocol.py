"""Execute the registered 8 + 36 training queue and locked test matrix.

Stages: pilot, core, select, test. `test` requires a verified text encoder.
Runs are resumable: existing run JSON/checkpoints are checked and skipped.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from experiment import (ALIGNED, DESIGN, ROOT, MODELS, data_bundle, load_model,
                        metric, predict, read_conditions, run_q0, save_norm,
                        sha_file, train_one)


def manifest():
    with (DESIGN / "训练运行清单.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def verify_config(args):
    if args.source == "verified" and not args.text_cache:
        raise ValueError("Verified route requires --text-cache")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.source == "precomputed" and args.out.resolve() == (ROOT / "outputs" / "official").resolve():
        raise ValueError("Use a distinct --out directory for provisional precomputed runs")


def cached_run(args, row, data, norm, hidden, lr, dropout, teacher=None):
    run_id = row["run_id"]
    run_path = args.out / "runs" / f"{run_id}.json"
    if run_path.exists():
        record = json.loads(run_path.read_text(encoding="utf-8"))
        if (record["model"] != row["model"] or record["seed"] != row["seed"] or record["source"] != args.source
                or record["hidden_dim"] != hidden or record["learning_rate"] != lr or record["dropout"] != dropout):
            raise ValueError(f"Existing run conflicts: {run_id}")
        if not Path(record["checkpoint"]).exists(): raise ValueError(f"Missing checkpoint: {run_id}")
        print(f"skip {run_id}", flush=True)
        return record
    return train_one(args, data, norm, row["model"], row["seed"], hidden, lr, dropout, run_id, teacher)


def run_pilot(args):
    data, norm = data_bundle(args.source, args.text_cache)
    save_norm(args.out, norm, args.source, args.text_cache)
    run_q0(args)
    records = []
    for row in manifest():
        if row["stage"] != "pilot": continue
        records.append(cached_run(args, row, data, norm, row["hidden_dim"], row["learning_rate"], row["dropout"]))
    records.sort(key=lambda r: (-r["best_score"], r["hidden_dim"], r["learning_rate"], r["dropout"]))
    winner = {k: records[0][k] for k in ("hidden_dim", "learning_rate", "dropout")}
    winner["source"] = args.source
    (args.out / "pilot_choice.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")
    print(json.dumps({"pilot_choice": winner, "runs": len(records)}, ensure_ascii=False))


def run_core(args):
    choice = json.loads((args.out / "pilot_choice.json").read_text(encoding="utf-8"))
    if choice["source"] != args.source: raise ValueError("Pilot source mismatch")
    data, norm = data_bundle(args.source, args.text_cache)
    save_norm(args.out, norm, args.source, args.text_cache)
    rows = [r for r in manifest() if r["stage"] != "pilot"]
    # F01 must precede D1 for all paired seeds. The manifest is not assumed
    # to be sorted in dependency order.
    rows.sort(key=lambda r: (r["model"] == "D1", r["run_id"]))
    for row in rows:
        teacher = None
        if row["model"] == "D1":
            match = [r for r in rows if r["model"] == "F01" and r["seed"] == row["seed"]]
            if len(match) != 1: raise ValueError(f"No unique F01 teacher for {row['run_id']}")
            teacher = args.out / "checkpoints" / f"{match[0]['run_id']}.pt"
        cached_run(args, row, data, norm, choice["hidden_dim"], choice["learning_rate"], choice["dropout"], teacher)


def run_select(args):
    rows = [r for r in manifest() if r["stage"] != "pilot"]
    results = {}
    data, norm = data_bundle(args.source, args.text_cache)
    valid = data["valid"]
    for row in rows:
        path = args.out / "runs" / f"{row['run_id']}.json"
        if not path.exists(): raise ValueError(f"Incomplete core run: {path.name}")
        rec = json.loads(path.read_text(encoding="utf-8"))
        cp = Path(rec["checkpoint"])
        model, info = load_model(cp, norm, args.device)
        if info["source"] != args.source: raise ValueError("Checkpoint source mismatch")
        p, r, _, _ = predict(model, valid, read_conditions("V-select")[0], args.batch_size, args.device)
        results[row["run_id"]] = {"model": row["model"], "seed": row["seed"], "score": rec["best_score"],
                                  "clean": metric(valid.y_cls, valid.y_reg, p, r), "checkpoint": str(cp)}
    grouped = {}
    for rec in results.values(): grouped.setdefault(rec["model"], []).append(rec)
    for model_id, group in grouped.items():
        if len(group) != 3: raise ValueError(f"Need 3 seeds for {model_id}")
    ref_f1 = np.mean([r["clean"]["macro_f1"] for r in grouped["F00"]])
    ref_mae = np.mean([r["clean"]["mae"] for r in grouped["F00"]])
    candidates = []
    for model_id in ("F00", "F01", "F10", "F11", "F20", "F21", "D1"):
        group = grouped[model_id]
        f1 = np.mean([r["clean"]["macro_f1"] for r in group]); mae = np.mean([r["clean"]["mae"] for r in group])
        if model_id != "F00" and (f1 < ref_f1 - 0.02 or mae > ref_mae + 0.10): continue
        candidates.append((float(np.mean([r["score"] for r in group])), model_id))
    if not candidates: raise ValueError("No qualifying model, including F00")
    candidates.sort(reverse=True)
    best_model = candidates[0][1]
    chosen = sorted(grouped[best_model], key=lambda r: r["score"])[1]
    lock = {"status": "locked_verified" if args.source == "verified" else "provisional_only",
            "source": args.source, "selected_model": best_model, "selected_seed": chosen["seed"],
            "selected_checkpoint": chosen["checkpoint"], "all_valid_results": results,
            "condition_sha256": sha_file(DESIGN / "评估条件清单.jsonl"),
            "protocol_sha256": sha_file(DESIGN / "实验协议.json"),
            "data_sha256": sha_file(ALIGNED),
            "code_sha256": {p.name: sha_file(p) for p in Path(__file__).parent.glob("*.py")},
            "text_cache_sha256": sha_file(args.text_cache) if args.text_cache else None,
            "checkpoint_sha256": {Path(r["checkpoint"]).name: sha_file(Path(r["checkpoint"])) for r in results.values()}}
    path = args.out / "protocol_lock.json"
    path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"lock": str(path), "selected_model": best_model, "selected_seed": chosen["seed"], "status": lock["status"]}, ensure_ascii=False))


def check_lock(args, lock):
    if lock["status"] != "locked_verified": raise ValueError("Test requires a verified locked protocol")
    if lock["source"] != args.source: raise ValueError("Source changed after lock")
    checks = ((DESIGN / "评估条件清单.jsonl", lock["condition_sha256"]),
              (DESIGN / "实验协议.json", lock["protocol_sha256"]), (ALIGNED, lock["data_sha256"]))
    for path, digest in checks:
        if sha_file(path) != digest: raise ValueError(f"Changed after lock: {path}")
    for name, digest in lock["code_sha256"].items():
        if sha_file(Path(__file__).parent / name) != digest: raise ValueError(f"Code changed after lock: {name}")
    if sha_file(args.text_cache) != lock["text_cache_sha256"]: raise ValueError("Text cache changed after lock")
    for name, digest in lock["checkpoint_sha256"].items():
        if sha_file(args.out / "checkpoints" / name) != digest: raise ValueError(f"Checkpoint changed: {name}")


def run_test(args):
    lock = json.loads((args.out / "protocol_lock.json").read_text(encoding="utf-8"))
    check_lock(args, lock)
    data, norm = data_bundle(args.source, args.text_cache, ("train", "test"))
    test = data["test"]
    for run_id, rec in lock["all_valid_results"].items():
        model_id = rec["model"]
        model, _ = load_model(Path(rec["checkpoint"]), norm, args.device)
        suites = ["E-main"] + (["E-random"] if model_id in ("F11", "F20", "F21", "D1") else [])
        for suite in suites:
            metric_path = args.out / "metrics" / f"{run_id}_test_{suite}.json"
            pred_path = args.out / "predictions" / f"{run_id}_test_{suite}.csv.gz"
            if metric_path.exists() and pred_path.exists():
                print(f"skip {run_id} {suite}", flush=True); continue
            metric_path.parent.mkdir(parents=True, exist_ok=True); pred_path.parent.mkdir(parents=True, exist_ok=True)
            summary = []
            with gzip.open(pred_path, "wt", newline="", encoding="utf-8") as f:
                fields = ["condition_id", "sample_id", "video_id", "true_polarity", "true_intensity",
                          "p_negative", "p_neutral", "p_positive", "pred_polarity", "pred_intensity", "fallback", "mask"]
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
                for cond in read_conditions(suite):
                    p, r, fb, rows = predict(model, test, cond, args.batch_size, args.device, rows=True)
                    m = metric(test.y_cls, test.y_reg, p, r)
                    m["condition_id"] = cond["condition_id"]; m["fallback_rate"] = float(fb.mean())
                    m["n_actually_affected"] = sum(sum(row["mask"]["actual_removed"].values()) > 0 for row in rows)
                    realized = [row["mask"]["actual_rate"][k] for row in rows for k in cond["modalities"]
                                if row["mask"]["actual_rate"][k] is not None]
                    m["realized_rate_mean_targeted_modalities"] = float(np.mean(realized)) if realized else None
                    summary.append(m)
                    for j, row in enumerate(rows):
                        writer.writerow({"condition_id": cond["condition_id"], "sample_id": row["sample_id"],
                            "video_id": row["sample_id"].split("$_$")[0], "true_polarity": int(test.y_cls[j]),
                            "true_intensity": float(test.y_reg[j]), "p_negative": row["p_negative"],
                            "p_neutral": row["p_neutral"], "p_positive": row["p_positive"],
                            "pred_polarity": row["pred_polarity"], "pred_intensity": row["pred_intensity"],
                            "fallback": row["fallback"], "mask": json.dumps(row["mask"], ensure_ascii=False)})
                    print(f"{run_id} {suite} {len(summary)}", flush=True)
            metric_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["pilot", "core", "select", "test"])
    ap.add_argument("--source", choices=["verified", "precomputed"], default="verified")
    ap.add_argument("--text-cache", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "official")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--min-epochs", type=int, default=10)
    args = ap.parse_args(); verify_config(args)
    torch.set_num_threads(args.threads)
    {"pilot": run_pilot, "core": run_core, "select": run_select, "test": run_test}[args.stage](args)


if __name__ == "__main__": main()
