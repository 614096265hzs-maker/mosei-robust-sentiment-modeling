# 复杂场景下多模态情感识别建模

本仓库基于[原 CLMER 项目](https://github.com/HPUhushicheng/MSA-Challenge-The-4th-Pazhou-AI-Competition)改造，面向本题的三个问题：原视频输入构建、缺失模态下的情感极性与强度预测、可核验的模型解释。

请从[中文建模说明](README-modeling_zh.md)开始。实际执行代码位于 `modeling/`，预先登记的实验方案与清单位于 `protocol/`。原项目的[中文说明](docs/upstream/README_zh.md)与[英文说明](docs/upstream/README.md)单独存档，供核对来源；原项目报告的成绩不代表本题结果。

## 当前状态

- 已提供数据读取、文本编码器核验、Q1 视频处理、Q2 训练与测试调度、Q3 遮挡解释和配对统计的代码入口。
- 本仓库未运行正式训练和测试，也未填写实验指标。
- 运行前需自行准备赛题数据、相应模型权重和人工对齐标注；正式路线必须先核验文本编码器。
- 原始视频、特征文件、模型权重和运行输出不纳入仓库。

保留原项目的 Apache-2.0 许可，详见 [LICENSE](LICENSE)。来源与改造关系见[来源说明](docs/upstream/PROVENANCE.md)。
