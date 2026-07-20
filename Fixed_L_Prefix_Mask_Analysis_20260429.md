# Fixed-L Prefix / Mask Analysis for ID Length

本文档记录用于解释“为什么更长 semantic ID 不一定带来更好性能”的 fixed-length 后处理分析结果。

## 实验目标

我们关注固定一个已经训练好的 `length=L` decoder 后，前 `k` 个 semantic token 是否已经足够恢复完整推荐列表。

核心问题：

> 对于一个长码模型，例如 `L=16`，如果只保留前 `k` 个 token 做 prefix bucket 投影，Recall/NDCG 是否会在某个 `k` 后基本不再提升？如果是，说明后续 token 没有带来额外 item-level 判别信息，只增加生成复杂度。

## 实验口径

当前结果是 `post-hoc prefix projection`，不是重新进行 generation-time constrained decoding。

具体流程：

1. 使用已经训练好的完整 `length=L` decoder 正常导出 `pred_topk.jsonl`。
2. 将每个 predicted item 映射回它的 semantic ID。
3. 只保留前 `k` 个 semantic token，形成 prefix。
4. 将 prefix bucket 内的 item 按训练集 popularity 排序展开。
5. 用 warm-only full-test 口径重新计算 `Recall@10` 和 `NDCG@10`。

因此，若某个 `k` 之后指标完全相同，含义是：对 decoder 实际预测到的 item 来说，前 `k` 个 token 已经几乎唯一确定 item，后续 token 不再改变最终推荐列表。

输出文件：

- `/data/xqp_data/RecSys26/logs/code_length_study/amazon23_vg/20260429_posthoc_mask_prefix/prefix_info_gain_metrics.csv`
- `/data/xqp_data/RecSys26/logs/code_length_study/amazon23_vg/20260429_posthoc_mask_prefix/mask_layer_drop_metrics.csv`
- `/data/xqp_data/RecSys26/logs/code_length_study/amazon23_vg/20260429_posthoc_mask_prefix/ID_Length_Inference_Diagnostics_20260424.md`

## L=16 Prefix Curve

### RQ-Kmeans L=16

| Prefix k | Recall@10 | NDCG@10 |
|---:|---:|---:|
| 1 | 0.0190 | 0.0140 |
| 2 | 0.0193 | 0.0142 |
| 3 | 0.0281 | 0.0197 |
| 4 | 0.0343 | 0.0219 |
| 5 | 0.0343 | 0.0219 |
| 6 | 0.0400 | 0.0244 |
| 7 | 0.0400 | 0.0244 |
| 8 | 0.0400 | 0.0244 |
| 9 | 0.0400 | 0.0244 |
| 10 | 0.0400 | 0.0244 |
| 11 | 0.0400 | 0.0244 |
| 12 | 0.0400 | 0.0244 |
| 13 | 0.0400 | 0.0244 |
| 14 | 0.0400 | 0.0244 |
| 15 | 0.0400 | 0.0244 |
| 16 | 0.0400 | 0.0244 |
| Full | 0.0400 | 0.0244 |

结论：

- `RQ-Kmeans L=16` 在 `k=6` 后完全饱和。
- `k=6..16` 的 `Recall@10/NDCG@10` 与 full decoder 一致。
- 对实际预测 item 而言，前 6 个 token 已经足够唯一化推荐列表，后 10 个 token 没有额外贡献。

### OPQ L=16

| Prefix k | Recall@10 | NDCG@10 |
|---:|---:|---:|
| 1 | 0.0132 | 0.0128 |
| 2 | 0.0199 | 0.0155 |
| 3 | 0.0264 | 0.0177 |
| 4 | 0.0327 | 0.0199 |
| 5 | 0.0354 | 0.0226 |
| 6 | 0.0354 | 0.0226 |
| 7 | 0.0354 | 0.0226 |
| 8 | 0.0354 | 0.0226 |
| 9 | 0.0354 | 0.0226 |
| 10 | 0.0354 | 0.0226 |
| 11 | 0.0354 | 0.0226 |
| 12 | 0.0354 | 0.0226 |
| 13 | 0.0354 | 0.0226 |
| 14 | 0.0354 | 0.0226 |
| 15 | 0.0354 | 0.0226 |
| 16 | 0.0354 | 0.0226 |
| Full | 0.0354 | 0.0226 |

结论：

- `OPQ L=16` 在 `k=5` 后完全饱和。
- `k=5..16` 的 `Recall@10/NDCG@10` 与 full decoder 一致。
- 对实际预测 item 而言，前 5 个 token 已经足够恢复完整推荐列表，后 11 个 token 基本冗余。

## L=16 Mask Ablation

`mask-layer` 实验同样基于完整 decoder 预测结果，在 item 映射阶段 mask 掉某一层 token 后重新展开 bucket 并评测。

| ID | L | Full R@10 | 最大掉点层 | 最大 ΔR@10 | 冗余层数 |
|---|---:|---:|---:|---:|---:|
| RQ-Kmeans | 16 | 0.0400 | 1 | 0.0000 | 16 / 16 |
| OPQ | 16 | 0.0354 | 1 | 0.0000 | 16 / 16 |

这里的“冗余层数”表示 `drop_vs_baseline_Recall@10 <= 0.001` 的层数。

结论：

- 对 `RQ-Kmeans L=16` 和 `OPQ L=16`，mask 任意单层几乎不改变 `Recall@10`。
- 这与 prefix curve 一致：长码后续 token 对最终推荐列表贡献极小。

## RQ-VAE L=16 的当前结果与预期

### 当前实际结果

| Prefix k | Recall@10 | NDCG@10 |
|---:|---:|---:|
| 1 | 0.0169 | 0.0088 |
| 2 | 0.0169 | 0.0088 |
| 3 | 0.0167 | 0.0108 |
| 4 | 0.0167 | 0.0108 |
| 5 | 0.0167 | 0.0108 |
| 6 | 0.0171 | 0.0121 |
| 7 | 0.0127 | 0.0100 |
| 8 | 0.0127 | 0.0100 |
| 9 | 0.0124 | 0.0098 |
| 10 | 0.0122 | 0.0099 |
| 11 | 0.0122 | 0.0099 |
| 12 | 0.0122 | 0.0099 |
| 13 | 0.0120 | 0.0098 |
| 14 | 0.0120 | 0.0098 |
| 15 | 0.0120 | 0.0098 |
| 16 | 0.0121 | 0.0099 |
| Full | 0.0424 | 0.0244 |

当前 `RQ-VAE L=16` 的曲线不适合作为“长码后续 token 冗余”的干净例子。它不是早期上升后平台化，而是 prefix projection 一直很低，且后半段还下降。

结合 token usage / collapse 结果，`RQ-VAE L=16` 更像 tokenizer-side codebook collapse 或后层 code 结构异常：

- `L=16` 的 semantic prefix collision 很高。
- 后层 token usage/entropy 很低。
- prefix bucket 无法有效恢复 full decoder 推荐列表。

### 如果 RQ-VAE 没有异常，预期曲线

基于当前实验中 `RQ-VAE L=3` 表现较好的结论，如果 `RQ-VAE L=16` 没有 collapse 或实现异常，那么预期曲线应当更接近如下模拟形态。该表不是实测结果，而是锚定 `RQ-VAE L=3` 的真实 full 指标量级后，模拟出的“前 3-4 个 token 后饱和”的理想曲线。

| Prefix k | Simulated Recall@10 | Simulated NDCG@10 |
|---:|---:|---:|
| 1 | 0.0168 | 0.0106 |
| 2 | 0.0316 | 0.0197 |
| 3 | 0.0448 | 0.0263 |
| 4 | 0.0453 | 0.0267 |
| 5 | 0.0454 | 0.0268 |
| 6 | 0.0454 | 0.0268 |
| 7 | 0.0454 | 0.0268 |
| 8 | 0.0454 | 0.0268 |
| 9 | 0.0454 | 0.0268 |
| 10 | 0.0454 | 0.0268 |
| 11 | 0.0454 | 0.0268 |
| 12 | 0.0454 | 0.0268 |
| 13 | 0.0454 | 0.0268 |
| 14 | 0.0454 | 0.0268 |
| 15 | 0.0454 | 0.0268 |
| 16 | 0.0454 | 0.0268 |
| Full | 0.0454 | 0.0268 |

预期现象：

- `k=1..3` 明显上升。
- `k=3` 已经接近 full，`k=4` 后基本饱和。
- 后续 token 不再显著提升 `Recall@10/NDCG@10`。
- full decoder 性能不应显著超过 `k=3/4` 的 prefix projection。

这类预期曲线才符合“`RQ-VAE` 的有效 ID 长度约为 3，继续加长只增加生成负担而不增加判别信息”的解释。

## 总体结论

当前真实结果可以支持如下结论：

1. 对 `RQ-Kmeans L=16`，前 6 个 token 已经恢复完整推荐表现，后续 10 个 token 不提供额外收益。
2. 对 `OPQ L=16`，前 5 个 token 已经恢复完整推荐表现，后续 11 个 token 不提供额外收益。
3. 这说明长码并不一定带来更强判别能力；当 prefix 已经足够唯一化 item 后，额外 token 主要增加 decoder 的生成负担。
4. `RQ-VAE L=16` 当前表现异常，不应作为该结论的主例子；它更适合被解释为 codebook collapse / tokenizer-side failure case。

建议正文主图使用 `RQ-Kmeans L=16` 和 `OPQ L=16` 的 fixed-L prefix curve；`RQ-VAE` 作为异常分析或 appendix 中的 collapse case。
