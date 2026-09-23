"""Join Q3 occlusion results with the separately audited attachment-4 map.

Only text segments with full approximate token timing receive time bounds.
Audio/visual segments retain feature indices until their extraction mapping is
independently verified. This script never invents exact seconds or labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiment import ROOT


OVERLAP = {"03", "07", "08", "12", "15"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--explanations", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "official" / "explanations")
    args = ap.parse_args()
    explanations = json.loads(args.explanations.read_text(encoding="utf-8"))
    mapping = {x["sample_id"]: x for x in json.loads(args.mapping.read_text(encoding="utf-8"))}
    if len(explanations) != 20 or len(mapping) != 20: raise ValueError("Attachment 4 requires 20 records")
    cards = []
    for record in explanations:
        sample_id = record["sample_id"]
        if sample_id not in mapping: raise ValueError(f"Missing mapping: {sample_id}")
        word_map = {x["feature_position"]: x for x in mapping[sample_id]["token_positions"]}
        evidence = []
        for modality, segments in record["top20_percent_segments_exclusive_end"].items():
            for start, end in segments:
                item = {"modality": modality, "feature_start": start, "feature_end_exclusive": end,
                        "text": None, "start_seconds": None, "end_seconds": None,
                        "mapping_status": "unavailable"}
                if modality == "T":
                    tokens = [word_map.get(i) for i in range(start, end)]
                    if tokens and all(x is not None and x["status"] == "approximate" for x in tokens):
                        words = []
                        for token in tokens:
                            if not words or words[-1] != token["word"]: words.append(token["word"])
                        item.update({"text": " ".join(words),
                                     "start_seconds": min(x["start_seconds"] for x in tokens),
                                     "end_seconds": max(x["end_seconds"] for x in tokens),
                                     "mapping_status": "approximate"})
                evidence.append(item)
        cards.append({"sample_id": sample_id, "pred_polarity": record["pred_polarity"],
                      "pred_intensity": record["pred_intensity"], "probabilities": record["probabilities"],
                      "main_modality": record["main_modality"],
                      "modality_sensitivity": record["modality_sensitivity"],
                      "evidence": evidence, "test_overlap_disclosure": sample_id in OVERLAP,
                      "reference_label": None, "mapping_note": "Times are approximate text alignment, not official A/V feature timestamps"})
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "attachment4_explanation_cards.json"
    json_path.write_text(json.dumps(cards, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    csv_path = args.out / "attachment4_explanation_cards.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["sample_id", "pred_polarity", "pred_intensity", "main_modality", "probabilities",
                  "modality_sensitivity", "evidence", "test_overlap_disclosure"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for card in cards:
            writer.writerow({key: json.dumps(card[key], ensure_ascii=False) if key in
                             ("probabilities", "modality_sensitivity", "evidence") else card[key] for key in fields})
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(cards)}, ensure_ascii=False))


if __name__ == "__main__": main()
