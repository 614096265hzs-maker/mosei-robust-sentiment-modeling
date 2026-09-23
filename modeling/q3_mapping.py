"""为附件4构建可审计的文本词元与视频时间近似映射。

匹配的词元编号确定词元与词语的对应关系；强制对齐
提供词语近似时间，但无法核验官方音频、视觉
特征的提取时间，因此这些模态仍标记为 `unavailable`。
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from experiment import A4, ROOT, load_pickle
from q1_baseline import mp4_duration
from q1_pipeline import align_words, extract_wav, map_words


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bert", type=Path, required=True, help="本地 BERT 分词器目录")
    ap.add_argument("--device", default="cpu", help="计算设备")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "q3_mapping",
                    help="附件4映射输出目录")
    args = ap.parse_args()
    try:
        import whisperx
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("请先安装 requirements-q1.txt") from error
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.bert), use_fast=True, local_files_only=True)
    aligner, metadata = whisperx.load_align_model(language_code="en", device=args.device)
    reports = []
    for number in range(1, 21):
        name = f"{number:02d}"
        obj = load_pickle(A4 / f"{name}.pkl")
        video = A4 / "videos" / f"{name}.mp4"
        report = {"sample_id": name, "video_path": str(video),
                  "mapping_status": {"text": "unavailable", "audio": "unavailable", "vision": "unavailable"},
                  "token_positions": [], "verification_note": "official audio/visual feature timestamps not supplied"}
        try:
            duration = mp4_duration(video)
            words = str(obj["raw_text"]).split()
            encoded = tokenizer(words, is_split_into_words=True, padding="max_length",
                                truncation=True, max_length=50, return_tensors="np")
            bert = np.asarray(obj["text_bert"])
            fields = ("input_ids", "attention_mask", "token_type_ids")
            if any(key not in encoded for key in fields): raise ValueError("分词器结果缺少 BERT 输入字段")
            reconstructed = np.stack([encoded[key][0] for key in fields])
            if not np.array_equal(reconstructed, bert): raise ValueError("BERT 词元编号或掩码与附件不一致")
            descriptor, temporary = tempfile.mkstemp(suffix=".wav")
            os.close(descriptor)
            wav = Path(temporary)
            try:
                extract_wav(video, wav)
                aligned = align_words(wav, str(obj["raw_text"]), duration, aligner, metadata, args.device)
            finally:
                wav.unlink(missing_ok=True)  # 只删除这一明确路径的临时文件。
            intervals = map_words(words, aligned, duration)
            ids = encoded.word_ids()
            for pos, word_id in enumerate(ids):
                if word_id is None or not bert[1, pos]: continue
                interval = intervals[word_id]
                report["token_positions"].append({"feature_position": pos, "word_index": word_id,
                    "word": words[word_id], "start_seconds": interval[0] if interval else None,
                    "end_seconds": interval[1] if interval else None,
                    "status": "approximate" if interval else "unavailable"})
            valid_times = [x for x in report["token_positions"] if x["status"] == "approximate"]
            if valid_times: report["mapping_status"]["text"] = "approximate"
            report["duration_seconds"] = duration
            report["coverage"] = len(valid_times) / max(len(report["token_positions"]), 1)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        reports.append(report)
        print(f"{name} {report['mapping_status']['text']}", flush=True)
    path = args.out / "attachment4_text_mapping.json"
    path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "rows": len(reports)}, ensure_ascii=False))


if __name__ == "__main__": main()
