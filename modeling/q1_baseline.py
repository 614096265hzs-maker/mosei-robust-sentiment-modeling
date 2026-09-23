"""Freeze the 100-video inventory and the Q1-U uniform-time baseline.

This does not perform forced alignment or extract acoustic/visual features.
Those stages require a verified aligner and feature extractors.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import struct
from pathlib import Path

import openpyxl

from experiment import DATA, ROOT


VIDEO_ROOT = DATA / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"


def mp4_duration(path: Path) -> float:
    duration = None
    with path.open("rb") as handle:
        def walk(start, end):
            nonlocal duration
            cursor = start
            while cursor + 8 <= end:
                handle.seek(cursor)
                size, kind = struct.unpack(">I4s", handle.read(8))
                header = 8
                if size == 1:
                    size = struct.unpack(">Q", handle.read(8))[0]
                    header = 16
                if size == 0: size = end - cursor
                if size < header or cursor + size > end: break
                payload = cursor + header
                if kind in (b"moov", b"trak", b"mdia"):
                    walk(payload, cursor + size)
                elif kind == b"mvhd":
                    handle.seek(payload)
                    version = handle.read(1)[0]
                    handle.seek(payload + (20 if version == 1 else 12))
                    scale = struct.unpack(">I", handle.read(4))[0]
                    amount = struct.unpack(">Q" if version == 1 else ">I", handle.read(8 if version == 1 else 4))[0]
                    duration = amount / scale if scale else None
                cursor += size
        walk(0, path.stat().st_size)
    if duration is None or duration <= 0: raise ValueError(f"Cannot read MP4 duration: {path}")
    return float(duration)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "q1")
    args = ap.parse_args()
    wb = openpyxl.load_workbook(VIDEO_ROOT / "label-100.xlsx", read_only=True, data_only=True)
    rows = list(wb["label"].values)
    fields = {v: i for i, v in enumerate(rows[0])}
    records = []
    for row in rows[1:]:
        video_id = str(row[fields["video_id"]]); clip_id = str(row[fields["clip_id"]])
        path = VIDEO_ROOT / video_id / f"{clip_id}.mp4"
        if not path.exists(): raise FileNotFoundError(path)
        duration = mp4_duration(path)
        raw_text = str(row[fields["text"]])
        words = re.findall(r"\S+", raw_text)
        intervals = [{"word_index": i, "word": word, "start": duration * i / len(words),
                      "end": duration * (i + 1) / len(words)} for i, word in enumerate(words)] if words else []
        records.append({"sample_id": f"{video_id}$_${clip_id}", "video_id": video_id,
                        "clip_id": clip_id, "video_path": str(path), "duration": duration,
                        "raw_text": raw_text, "word_count": len(words), "q1_u_intervals": intervals,
                        "q1_u_status": "uniform_approximation", "forced_alignment_status": "not_run",
                        "audio_feature_status": "not_run", "vision_feature_status": "not_run"})
    if len(records) != 100 or len({r["sample_id"] for r in records}) != 100:
        raise ValueError("Expected 100 unique source videos")
    # Freeze the evaluation audit before checking any alignment output.
    sorted_records = sorted(records, key=lambda r: (r["duration"], r["sample_id"]))
    strata = [sorted_records[:33], sorted_records[33:67], sorted_records[67:]]
    rng = random.Random(20260923)
    selected = []; used_videos = set()
    for label, group, quota in zip(("short", "medium", "long"), strata, (7, 6, 7)):
        shuffled = group.copy(); rng.shuffle(shuffled)
        chosen = [r for r in shuffled if r["video_id"] not in used_videos][:quota]
        if len(chosen) < quota:
            chosen.extend([r for r in shuffled if r not in chosen][:quota - len(chosen)])
        if len(chosen) != quota: raise ValueError("Audit stratum underfilled")
        for r in chosen: used_videos.add(r["video_id"])
        selected.extend({"sample_id": r["sample_id"], "video_id": r["video_id"],
                         "duration": r["duration"], "stratum": label} for r in chosen)
    rng.shuffle(selected)
    for i, item in enumerate(selected): item["role"] = "development" if i < 5 else "blind_quality"
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "q1_u_100.jsonl").open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.out / "frozen_audit_20.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "audit": len(selected), "quality": sum(x["role"] == "blind_quality" for x in selected),
                      "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__": main()
