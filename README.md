# LLM-Sentiment-Analysis

中文外卖评论「隐晦侮辱识别」多技术路线比较研究——**代码仓库**。

研究文稿与数据集不公开（数据合规，见下）；本仓库只包含分析脚本与合成示例数据。

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
scripts/            分析脚本（Jupyter notebook + Python）
examples/           合成示例数据（与真实用户无关，仅示意字段口径）
```

## 复现

按 `scripts/` 中 notebook 的顺序运行即可完整复现全部数值：

1. `dl_sentiment.ipynb` — char-BiGRU 训练与五种子集成
2. `dl_multi_seed.py` — 多种子批量训练
3. `tfidf_features.ipynb` — 经典 ML 特征工程
4. `llm_api_prompt_annotation.ipynb` — LLM API 标注（基础/增强提示）
5. `llm_metrics.ipynb` — LLM 方案评估
6. `method_comparison.ipynb` — 全方案统计比较（Bootstrap / Cochran's Q / McNemar+Holm）
7. `fleiss_kappa.ipynb` — 标注信度

## 数据可用性

- 研究语料来自某本地生活服务平台，经提供方去标识化后用于学术研究，
  依数据使用约定**不可再分发**，故不在本仓库公开。
- 依据学界通行惯例，可通过合理请求向作者获取（仅供同行评审与学术复核）。
- `examples/synthetic_sample.csv` 为人工撰写的合成样例，字段口径与真实
  语料一致，用于展示数据结构。
- 本仓库的 [LICENSE](LICENSE) 仅覆盖代码，不覆盖任何语料或衍生数据。

## 数值可信度声明

全部数值由分析脚本从逐条预测与金标准重算，无手工转录；统计推断
（Bootstrap CI、Cochran's Q、McNemar + Holm 校正、配对 ΔF1 置信区间）
均在 `method_comparison.ipynb` 中可复现。
