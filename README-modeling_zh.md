# 复杂场景下多模态情感识别：本题适配方案

本目录将[原 CLMER 仓库](https://github.com/HPUhushicheng/MSA-Challenge-The-4th-Pazhou-AI-Competition)的原视频处理、多模态融合和解释思路，改造为本题的**可追溯输入、局部缺失下预测、可核验解释**三条任务线。原仓库代码和文档保留作参考；本题执行入口是 `modeling/` 和 `protocol/`。沿用原 Apache-2.0 许可，见 `LICENSE`。原仓库自述的跨语言成绩不是本题成绩。

> 本仓库是代码与预注册实验方案。按使用者要求，上传前不运行模型训练，也不填写任何虚构指标。首次运行须确认环境、文本编码器、人工对齐标注和附件4时间映射。

## 问题、输入与交付

| 问题 | 本题实现 | 可核验交付 |
|---|---|---|
| Q1 视频到多模态输入 | 100条视频清单与均匀时间基线；给定转写的 WhisperX 强制对齐；独立定义的 BERT、16维语音和10维画面特征；词到50位置映射 | 100条状态记录、变长词区间、特征包、20条冻结人工审计抽样、15条质量评价程序 |
| Q2 情感极性与强度 | 官方对齐版特征，保留 train/valid/test；掩码感知三路编码器；三分类+[-3,3]回归；连续/独立位置缺失与门控、等权对照 | 8次预搜索+36次正式训练调度、15/85/140条件评价、30条附件3预测接口、视频级配对 bootstrap |
| Q3 模型解释 | 整模态和局部特征位置遮挡，另存内部门控系数；删除与随机对照；附件4文本位置近似时间映射 | 20条附件4预测/索引级解释、模态可用性和映射状态；未核验时不输出“精确秒数” |

完整实验条件、比较假设、选模规则和统计判据见[实验方案](protocol/完整实验方案.md)。原始仓库主要是跨语言分类，不具备本题需要的强度回归、50位置特征接口与缺失矩阵；其 R1-Omni、emotion2vec、SigLIP 等实现未被直接拼接到附件2的 768/74/35 维特征上。Q1自建音视频特征也**不声称**与附件2同维同源。

## 数据放置与保护

此仓库不上传原始 MP4、PKL、Excel、模型权重或预测结果。将本题 `E题数据/` 放在仓库外的本地目录，并设置：

```powershell
$env:MSA_DATA_ROOT = 'D:\data\mosei_problem'
```

程序将从 `$env:MSA_DATA_ROOT\E题数据\...` 读取附件1—4。若数据放在仓库根目录，变量可省略。`.gitignore` 阻止常见原始数据、检查点和输出进入新提交；提交前仍需人工复核 `git status`。

附件2固定官方 train/valid/test（3395/728/727）；附件3 30 条和附件4 20 条只用于最终专项输出。附件4 的 03、07、08、12、15 与附件2 test 重合，不能把附件4称作完全独立测试集。附件4 的13号对齐视觉为全零，保留样本并将视觉记为不可用。

## 文本编码器的硬门槛

附件3只有 `text_bert`，没有附件2的768维 `text`。原仓库提到 `bert-base-uncased`，可把它作为**候选**，但名称无法证明它生成了附件2的预计算向量。先将候选模型权重放在本地，安装 `requirements-modeling.txt`，再运行：

```powershell
python modeling/experiment.py encode --encoder 'D:\models\bert-base-uncased' --text-cache outputs/official/verified_text.npz
```

脚本用相同词元输入重算附件2 train/valid/test 与附件4全部20条，在有效位置和原 `text` 逐值计算 MAE；超出阈值即失败，不生成“verified”缓存。检查点文件逐项计算 SHA-256。正式路线还需要核对词元与原文的一致性以及 BERT 权重来源。若候选不匹配，须寻找真正的编码器或重新定义并重训所有训练/专项文本表示，不能混用两套文本空间。

## Q1：100条原视频

需要本地 `ffmpeg`、`requirements-q1.txt`、候选 BERT 与 WhisperX 对齐模型。先生成统一基线并冻结审计样本，再执行提取：

```powershell
python modeling/q1_baseline.py
python modeling/q1_pipeline.py --bert 'D:\models\bert-base-uncased'
python modeling/q1_evaluate.py --annotations 'D:\annotations\q1_words.json'
```

`q1_evaluate.py` 需人工词起止标注，未提供时不能计算边界误差。失败或部分成功样本仍留在100条记录。Q1自建的音视频特征定义在 `q1_pipeline.py`；它们仅用于 Q1 的输入构造与质量分析，不能直接送进 Q2 已训练的74/35维编码器。

## Q2：固定实验队列

正式路线必须先通过 `encode`。队列按[机器可读清单](protocol/训练运行清单.jsonl)执行；F01同种子检查点是 D1 教师。验证集早停与选型后，`select` 锁定数据、代码、条件、缓存和全部检查点哈希；`test` 检查锁后才读取测试结果。

```powershell
python modeling/run_protocol.py pilot  --text-cache outputs/official/verified_text.npz
python modeling/run_protocol.py core   --text-cache outputs/official/verified_text.npz
python modeling/run_protocol.py select --text-cache outputs/official/verified_text.npz
python modeling/run_protocol.py test   --text-cache outputs/official/verified_text.npz
python modeling/paired_statistics.py
```

设备默认为CPU；可用 `--device cuda`、`--threads`、`--batch-size` 指定资源。最大50轮、最少10轮、早停耐心8轮。44次训练和数百万次条件推理有较高计算成本，先做 Q0 与单轮测速，再规划资源。`--source precomputed` 仅是附件2/4代码调试路线，不能生成附件3预测，也不能解除正式测试锁。

附件3与附件4的专项输出：

```powershell
python modeling/experiment.py special --source verified --text-cache outputs/official/verified_text.npz --checkpoint 'outputs/official/checkpoints/选定检查点.pt' --kind a3
python modeling/experiment.py special --source verified --text-cache outputs/official/verified_text.npz --checkpoint 'outputs/official/checkpoints/选定检查点.pt' --kind a4
```

CSV使用内部字段 `sample_id,pred_polarity,pred_intensity,p_negative,p_neutral,p_positive,...`。若主办方发布正式提交模板，须再做字段映射核对。

## Q3：解释及视频回映

```powershell
python modeling/explain.py --source verified --text-cache outputs/official/verified_text.npz --checkpoint 'outputs/official/checkpoints/选定检查点.pt' --split a4
python modeling/q3_mapping.py --bert 'D:\models\bert-base-uncased'
python modeling/export_explanation_cards.py --explanations 'outputs/official/explanations/选定检查点_a4_w3.json' --mapping 'outputs/q3_mapping/attachment4_text_mapping.json'
```

解释分开存储门控系数与遮挡敏感度：门控不是因果贡献。`explain.py` 在预计算特征层做 P-F 遮挡，默认3位置窗口，并输出原始类别支持方向。当前随机删除对照匹配模态位置数，**尚未匹配连续段长度**；完整忠实度结论还需依照协议加入连续段匹配、保留曲线和噪声稳定性试验。

`q3_mapping.py` 只有在官方 `text_bert` 与本地 tokenizer 的50个 ID、mask完全一致时，才将文本词元对应到强制对齐词区间；时间仍标为 `approximate`。官方音频/视觉特征未附逐帧时间戳，映射状态保持 `unavailable`。需要逐条视频回看与人工核验后，才能把某个具体片段称为可靠的秒级证据。

## 复现与未完成项

- 代码已经组织成可运行入口，但上传版本未执行本题正式实验。所有指标应由实际运行生成；不要引用原仓库的竞赛分数作为本题结果。
- 原仓库权重未随 Git 仓库提供，`models/readme.md` 与 `R1-Omni/readme.md` 也说明权重需另取。上传代码不等于提供可离线运行的大模型。
- `protocol/` 中 P-T 编码前文本缺失、零值规则敏感性、完整解释忠实度和人工映射审计属于尚待实现/执行的补充项。论文中须如实标明。
- 公开提交若有50MB附件限制，应只打包必要轻量模型、代码与配置；原始赛题附件和大型 BERT 权重不能混入提交包。
