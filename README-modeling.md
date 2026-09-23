# Adapted multimodal sentiment modeling project

The adapted workflow for the Chinese modeling problem is documented in [中文说明](README-modeling_zh.md). It adds `modeling/` for Q1 feature extraction, Q2 missing-modality classification and intensity regression, and Q3 occlusion explanations, plus a preregistered `protocol/` directory.

The original CLMER cross-lingual code remains for attribution and reference. Its reported scores and raw-video model weights are not results or dependencies of the adapted 50-position experiment. No competition data, pretrained weights, or generated predictions are included. The adapted code has not been formally trained or evaluated in this repository.

Set `MSA_DATA_ROOT` to the local directory containing `E题数据/`; see the Chinese guide for commands, model verification, experimental boundaries, and remaining validation work.

