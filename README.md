# LLM-Sentiment-Analysis

中文外卖评论「隐晦侮辱识别」多技术路线比较研究——**代码仓库**。

研究文稿与数据集不公开（数据合规，见下文「数据可用性」）；本仓库只包含分析脚本与合成示例数据。

## 研究概要

在 420 条中文外卖评论（好评 182 / 正常差评 114 / 语气型 73 / 声音型 30 / 视觉型 21）
上，对十一条判定方案做双层评估：

- **判别层**：经典机器学习 4 条 + 深度网络 1 条（char-BiGRU 五种子软集成）+
  大语言模型 6 条（MiMo / DeepSeek / Qwen × 基础 / 增强提示），
  以金标准人工标注为参照。
- **价值层**：以文本审核平台的真实误判代价结构为参照，做阈值敏感性与
  决策成本分析。

核心发现：在类别高度不平衡（59.0% 为好评）的任务上，准确率不再是有效指标——
常数分类器准确率 0.918，可击败十一条真实方案中的十条。

## 仓库结构

```
scripts/
├── tfidf_features.ipynb              # 经典 ML：TF-IDF 特征 + 四种传统模型
├── dl_sentiment.ipynb                # 深度学习：char-BiGRU 训练与评估
├── dl_multi_seed.py                  # 深度学习：五种子批量训练脚本
├── llm_api_prompt_annotation.ipynb   # LLM：API 标注（基础/增强两套提示词）
├── llm_metrics.ipynb                 # LLM：方案评估指标计算
├── method_comparison.ipynb           # 全方案统计比较（Bootstrap / Cochran's Q / McNemar+Holm）
└── fleiss_kappa.ipynb                # 标注信度（Fleiss' κ）
examples/
└── synthetic_sample.csv              # 合成样例（与真实用户无关，仅示意字段口径）
```

## 环境要求

- Python 3.10+
- 依赖：pandas / scikit-learn / torch / openai（LLM API 调用）/ matplotlib
- LLM 标注脚本需要各厂商 API Key，通过环境变量注入，勿写入代码

## 复现顺序

1. `tfidf_features.ipynb` — 经典 ML 基线
2. `dl_sentiment.ipynb` + `dl_multi_seed.py` — char-BiGRU 五种子集成
3. `llm_api_prompt_annotation.ipynb` — LLM 标注（基础/增强提示）
4. `llm_metrics.ipynb` — LLM 方案评估
5. `fleiss_kappa.ipynb` — 标注信度检验
6. `method_comparison.ipynb` — 全方案统计比较与价值层分析

## 数据可用性

- 研究语料来自某本地生活服务平台，经提供方去标识化后用于学术研究，
  依数据使用约定**不可再分发**，故不在本仓库公开。
- 依据学界通行惯例，可通过合理请求向作者获取（仅供同行评审与学术复核）。
- `examples/synthetic_sample.csv` 为人工撰写的合成样例，字段口径与真实
  语料一致（`类型 / 评论 / 负面性标签 / 侮辱性标签 / 备注`），用于展示数据结构。
- 语料目录（`Data/`、`Raw Data/`）与研究文稿（`Paper/`）已列入
  `.gitignore`，永不入库。

## 数值可信度声明

全部数值由分析脚本从逐条预测与金标准重算，无手工转录；统计推断
（Bootstrap CI、Cochran's Q、McNemar + Holm 校正、配对 ΔF1 置信区间）
均在 `method_comparison.ipynb` 中可复现。

## 许可

[LICENSE](LICENSE) 仅覆盖本仓库代码；任何语料或衍生数据不受该许可覆盖。
