"""Q1: forced alignment and independently defined video/audio/text features.

Requires ffmpeg plus requirements-q1.txt. The resulting features are self-built
and are deliberately not treated as attachment-2's 768/74/35 representation.
Every one of the 100 records receives a success/partial/failed status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from experiment import ROOT


def extract_wav(video: Path, wav: Path):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                    "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True, timeout=180)


def align_words(wav: Path, text: str, duration: float, aligner, metadata, device: str):
    import whisperx
    audio = whisperx.load_audio(str(wav))
    result = whisperx.align([{"start": 0.0, "end": duration, "text": text}], aligner,
                            metadata, audio, device, return_char_alignments=False)
    return [w for segment in result.get("segments", []) for w in segment.get("words", [])]


def map_words(expected: list[str], aligned: list[dict], duration: float):
    """Order-preserving exact normalized token match; unmatched words remain null."""
    import re
    from difflib import SequenceMatcher
    normalize = lambda s: re.sub(r"[^a-z0-9]+", "", str(s).lower())
    a = [normalize(w) for w in expected]
    b = [normalize(w.get("word", "")) for w in aligned]
    mapping = [None] * len(expected)
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)
    for op, i0, i1, j0, j1 in matcher.get_opcodes():
        if op != "equal": continue
        for i, j in zip(range(i0, i1), range(j0, j1)):
            start = aligned[j].get("start"); end = aligned[j].get("end")
            if start is not None and end is not None and 0 <= start < end <= duration + 0.05:
                mapping[i] = [max(0.0, float(start)), min(duration, float(end))]
    return mapping


def text_word_features(words: list[str], tokenizer, encoder, device: str):
    """Average subword hidden states. Chunks retain every source word."""
    parts = []
    with torch.no_grad():
        for offset in range(0, len(words), 64):
            chunk = words[offset:offset + 64]
            encoded = tokenizer(chunk, is_split_into_words=True, return_tensors="pt",
                                truncation=True, max_length=512)
            word_ids = encoded.word_ids()
            if max((i for i in word_ids if i is not None), default=-1) != len(chunk) - 1:
                raise ValueError("BERT truncation lost one or more words")
            input_dict = {k: v.to(device) for k, v in encoded.items()}
            hidden = encoder(**input_dict).last_hidden_state[0].cpu().numpy()
            for i in range(len(chunk)):
                positions = [j for j, word_id in enumerate(word_ids) if word_id == i]
                if not positions: raise ValueError(f"No subword for word {offset+i}")
                parts.append(hidden[positions].mean(axis=0).astype(np.float32))
    return np.stack(parts) if parts else np.empty((0, 768), np.float32)


def audio_frames(wav: Path):
    import librosa
    signal, sr = librosa.load(str(wav), sr=16000, mono=True)
    hop = 320; n_fft = 1024
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft, hop_length=hop)
    arr = np.vstack((mfcc, rms, zcr, centroid)).T.astype(np.float32)
    times = librosa.frames_to_time(np.arange(len(arr)), sr=sr, hop_length=hop)
    return times, arr  # 16 self-defined acoustic dimensions


def visual_frames(video: Path, sample_hz=5):
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f"Cannot decode video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0: raise ValueError("Invalid video FPS")
    step = max(1, round(fps / sample_hz))
    frame_id = 0; times = []; rows = []; source_ids = []; prev_gray = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame_id % step == 0:
                frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
                mean = rgb.mean(axis=(0, 1)); std = rgb.std(axis=(0, 1))
                edge = cv2.Canny(frame, 80, 160).mean() / 255
                motion = 0.0 if prev_gray is None else float(np.mean(np.abs(gray - prev_gray)))
                rows.append(np.r_[mean, std, gray.mean(), gray.std(), edge, motion])
                times.append(frame_id / fps); source_ids.append(frame_id)
                prev_gray = gray
            frame_id += 1
    finally: cap.release()
    if not rows: raise ValueError("No decoded video frames")
    return np.asarray(times), np.asarray(rows, np.float32), source_ids  # 10 visual dimensions


def pool_interval(times, features, start, end):
    selected = (times >= start) & (times < end)
    if not selected.any():
        midpoint = (start + end) / 2
        nearest = int(np.argmin(np.abs(times - midpoint)))
        if abs(times[nearest] - midpoint) <= max(0.05, (end - start) / 2): selected[nearest] = True
    return (features[selected].mean(axis=0), np.flatnonzero(selected).tolist()) if selected.any() else (None, [])


def to_50(words, intervals, text_features, audio_features, visual_features, frame_maps):
    count = len(words)
    size = min(50, count)
    if size == 0: raise ValueError("No words")
    groups = [[] for _ in range(size)]
    for i in range(count): groups[min(size - 1, i * size // count)].append(i)
    tx = np.zeros((50, text_features.shape[1]), np.float32)
    ax = np.zeros((50, audio_features.shape[1]), np.float32)
    vx = np.zeros((50, visual_features.shape[1]), np.float32)
    mask = np.zeros((50, 3), bool); mapping = []
    for slot, members in enumerate(groups):
        tx[slot] = text_features[members].mean(axis=0); mask[slot, 0] = True
        a = [audio_features[i] for i in members if np.isfinite(audio_features[i]).all()]
        v = [visual_features[i] for i in members if np.isfinite(visual_features[i]).all()]
        if a: ax[slot] = np.mean(a, axis=0); mask[slot, 1] = True
        if v: vx[slot] = np.mean(v, axis=0); mask[slot, 2] = True
        valid = [intervals[i] for i in members if intervals[i] is not None]
        mapping.append({"position": slot, "word_indices": members,
                        "words": [words[i] for i in members],
                        "start": min((x[0] for x in valid), default=None),
                        "end": max((x[1] for x in valid), default=None),
                        "source_frame_indices": sorted(set(j for i in members for j in frame_maps[i]))})
    return tx, ax, vx, mask, mapping


def process_one(record, tokenizer, encoder, aligner, align_meta, device):
    started = time.perf_counter()
    video = Path(record["video_path"])
    words = [item["word"] for item in record["q1_u_intervals"]]
    result = {"sample_id": record["sample_id"], "video_path": str(video),
              "duration": record["duration"], "word_count": len(words), "status": "failed",
              "feature_semantics": {"text": "BERT last layer subword mean, chunked",
                                    "audio": "13 MFCC + RMS + ZCR + spectral centroid",
                                    "visual": "RGB mean/std, gray mean/std, edge, frame difference"}}
    if not words: result["error"] = "no_words"; return result, None
    descriptor, name = tempfile.mkstemp(suffix=".wav")
    os.close(descriptor)
    wav = Path(name)
    try:
        extract_wav(video, wav)
        aligned = align_words(wav, record["raw_text"], record["duration"], aligner, align_meta, device)
        intervals = map_words(words, aligned, record["duration"])
        at, af = audio_frames(wav)
    finally:
        wav.unlink(missing_ok=True)  # One explicit temporary file.
    vt, vf, frame_ids = visual_frames(video)
    tf = text_word_features(words, tokenizer, encoder, device)
    a_rows = np.full((len(words), af.shape[1]), np.nan, np.float32)
    v_rows = np.full((len(words), vf.shape[1]), np.nan, np.float32)
    frame_maps = [[] for _ in words]
    for i, interval in enumerate(intervals):
        if interval is None: continue
        audio, _ = pool_interval(at, af, *interval)
        visual, indices = pool_interval(vt, vf, *interval)
        if audio is not None: a_rows[i] = audio
        if visual is not None: v_rows[i] = visual; frame_maps[i] = [frame_ids[j] for j in indices]
    tx, ax, vx, mask, mapping = to_50(words, intervals, tf, a_rows, v_rows, frame_maps)
    coverage = sum(x is not None for x in intervals) / len(words)
    result.update({"status": "success" if coverage == 1 else "partial", "aligned_word_coverage": coverage,
                   "audio_word_coverage": float(np.isfinite(a_rows).all(axis=1).mean()),
                   "visual_word_coverage": float(np.isfinite(v_rows).all(axis=1).mean()),
                   "word_intervals": intervals, "position_mapping": mapping,
                   "elapsed_seconds": time.perf_counter() - started})
    return result, {"text": tx, "audio": ax, "vision": vx, "mask": mask}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bert", type=Path, required=True, help="Local BERT model directory; its identity is recorded, not assumed official")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "q1")
    args = ap.parse_args()
    try:
        import whisperx
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install requirements-q1.txt first") from error
    args.out.mkdir(parents=True, exist_ok=True)
    inventory = args.out / "q1_u_100.jsonl"
    if not inventory.exists(): raise FileNotFoundError("Run q1_baseline.py first")
    records = [json.loads(line) for line in inventory.open(encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(str(args.bert), use_fast=True, local_files_only=True)
    encoder = AutoModel.from_pretrained(str(args.bert), local_files_only=True).eval().to(args.device)
    aligner, align_meta = whisperx.load_align_model(language_code="en", device=args.device)
    index_path = args.out / "q1_extraction_100.jsonl"
    with index_path.open("w", encoding="utf-8") as index:
        for number, record in enumerate(records, 1):
            try:
                result, arrays = process_one(record, tokenizer, encoder, aligner, align_meta, args.device)
                if arrays is not None:
                    filename = hashlib.sha256(record["sample_id"].encode()).hexdigest()[:20] + ".npz"
                    np.savez_compressed(args.out / filename, **arrays)
                    result["feature_file"] = filename
            except Exception as error:
                result = {"sample_id": record["sample_id"], "status": "failed",
                          "error": f"{type(error).__name__}: {error}"}
            index.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{number}/100 {record['sample_id']} {result['status']}", flush=True)
    config = {"bert_directory": str(args.bert), "device": args.device,
              "note": "Self-built features; no direct attachment-2 compatibility claim"}
    (args.out / "q1_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
