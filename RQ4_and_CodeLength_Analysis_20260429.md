# RQ4 距离可视化与 Code Length 诊断实验整理

日期：2026-04-29

本文档把原先分散记录在 `Process.md` 中的两个实验块单独整理出来：

- `RQ4`：基于距离与邻域结构的 SID item semantics 可视化与主表指标。
- `Code Length`：解释 semantic ID 码长 scaling 非单调现象的 inference-only 诊断实验。

## 1. RQ4：SID Geometry 与 Item Semantics

### 研究问题

`RQ4. How do different semantic ID designs affect the learning of item semantics?`

当前实验回答的是：不同 Semantic ID 设计最终得到的离散 SID 空间，是否能够保留原始 item semantic embedding 空间中的局部语义邻域。

这是一种 `representation-level evaluation`，不是把所有 tokenizer 强行放到同一个 source embedding space 下重新训练的 `tokenizer-only controlled ablation`。

### 实验协议

| 项目 | 设置 |
| --- | --- |
| 数据集 | `amazon23_vg` |
| Reference semantic space | `amazon23_vg.emb-t5-tdcb.npy` |
| Reference distance | cosine distance |
| SID distance | native SID distance |
| 主表指标 | `Jaccard@20`, `RBO@20 (p=0.9)` |
| 对比方法 | `I02-TIGER`, `I03-SEATER`, `I04-SemID`, `I05-LETTER`, `I06-DiffGRM`, `I07-RQKmeans` |
| 脚本 | `scripts/analyze_sid_geometry.py` |

`native SID distance` 的具体定义：

- `I02/I03/I04/I05/I07`：使用 prefix-based SID distance。
- `I06`：使用 Hamming distance。
- 对有 dedup suffix 的方法，在几何分析前去掉 dedup 后缀位。

### 主表结果

数值越高越好。

| Method | Jaccard@20 | RBO@20 |
| --- | ---: | ---: |
| I02-TIGER | 0.069283 | 0.101570 |
| I03-SEATER | 0.013739 | 0.018789 |
| I04-SemID | 0.026286 | 0.021477 |
| I05-LETTER | 0.038517 | 0.048360 |
| I06-DiffGRM | **0.172434** | 0.180635 |
| I07-RQKmeans | 0.128040 | **0.196671** |

### 结果解读

- `I06-DiffGRM` 的 `Jaccard@20` 最高，说明它最能保留 reference semantic space 中“哪些 item 应该进入局部邻域”这一结构。
- `I07-RQKmeans` 的 `RBO@20` 最高，说明它最能保留局部邻域中靠前 item 的排序一致性。
- `I02-TIGER` 是比较稳定的中间层方法。
- `I05-LETTER` 弱于 `TIGER`，但没有完全崩掉。
- `I03-SEATER` 和 `I04-SemID` 在当前 T5-reference 邻域保留指标下较弱。

适合正文使用的表述：

> 不同 SID 设计保留 item semantics 的能力并不相同。Diffusion/quantization-based 方法在局部语义几何保留上明显更强，但“邻域成员保留”和“邻域前排排序保留”不是完全相同的能力。

### Case Study 可视化

最终选择的代表性 anchor：

- `anchor_item_id = 18089`

选择依据：

- 避免只有 `I06/I07` 聚集在 target 周围、其它方法全部偏到一侧的极端 case。
- reference top-K 和 SID top-K 在二维投影中更展开。
- 能看出不同方法从内层到外层的还原度差距。
- 该 anchor 是在检查 `10471 / 792 / 18089 / 9906 / 14885` 等多个候选后选出的。

最终图目录：

- `tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/`

关键图：

![RQ4 case study](tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/rq4_case_study_selected.png)

![RQ4 neighborhood overview](tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/rq4_sid_geometry_neighbors.png)

可视化设置：

- 使用真实散点图，不使用径向示意图。
- 去掉 anchor-to-SID 连线。
- 去掉参考圆环。
- 使用紧凑 `2x3` panel 布局。
- 使用低饱和配色和更接近参考图的 marker 语义。
- 保留灰色 context points 的可见度，但过滤极端视窗外点，避免边界假象。
- 只对红/蓝/紫 top-K marker 做轻量 deterministic 防重叠处理；灰色 context 保持原始投影位置。

### 边界说明

这个 RQ4 实验回答的是：

- 各方法最终得到的 SID representation 是否保留 item semantic geometry。

它不直接回答：

- 如果把所有方法都强行放到同一个 source embedding space 重新训练 tokenizer，谁的量化器本身更强。

后者更适合作为 supplementary ablation，因为强行统一 source space 会改变部分方法定义：

- `SEATER` 如果改到 T5 source，会变得接近 content RQ tokenizer。
- `LETTER` 官方实现显式依赖 `cf_emb`，移除 CF 后不再是原始 LETTER。
- `SemID` 本质是 metadata/category hierarchy 的离散编码，不是 embedding quantizer。

## 2. Code Length 诊断实验

### 目标

解释 semantic ID 码长 scaling 为什么不是单调提升：

- `RQ-VAE` 和 `RQ-Kmeans` 往往短码长如 `L=2/3` 更强。
- `OPQ` 在 `L=6` 附近达到最好效果。
- 继续增加 code length 不一定提升推荐性能。

### 实验协议

| 项目 | 设置 |
| --- | --- |
| 数据集 | `amazon23_vg` |
| Decoder | main-table-aligned `t5-small` |
| 长度集合 | `L={2,3,4,6,8,12,16}` |
| 方法 | `RQ-VAE`, `RQ-Kmeans`, `OPQ` |
| 输入 run | `logs/code_length_study/amazon23_vg/20260421_clean_suffix_mainalign_t5small` |
| 脚本 | `scripts/length_inference_diagnostics.py` |

两个 inference-only 诊断：

- `Token utilization / Codebook collapse`：统计每个 length 下各层 codebook 的 token 使用频率分布，只报告 `Entropy`、`Effective usage`、`Gini`。
- `Layer-wise information gain`：基于已有 decoder predictions 做 prefix projection 和 mask-layer projection；不重新训练 decoder，也不重新跑 GPU beam search。

### Decoder 性能随长度变化

下表为 warm-only full-test 指标，来自已有 decoder exports。

| Method | L | R@5 | N@5 | R@10 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RQ-VAE | 2 | **0.0313** | **0.0219** | 0.0476 | 0.0276 |
| RQ-VAE | 3 | 0.0292 | 0.0212 | 0.0454 | 0.0268 |
| RQ-VAE | 4 | 0.0187 | 0.0157 | 0.0393 | 0.0227 |
| RQ-VAE | 6 | 0.0256 | 0.0187 | 0.0440 | 0.0248 |
| RQ-VAE | 8 | 0.0280 | 0.0214 | 0.0442 | 0.0270 |
| RQ-VAE | 12 | 0.0266 | 0.0194 | **0.0511** | **0.0277** |
| RQ-VAE | 16 | 0.0261 | 0.0188 | 0.0424 | 0.0244 |
| RQ-Kmeans | 2 | **0.0294** | 0.0200 | **0.0448** | 0.0252 |
| RQ-Kmeans | 3 | 0.0270 | **0.0217** | 0.0443 | **0.0276** |
| RQ-Kmeans | 4 | 0.0240 | 0.0188 | 0.0369 | 0.0233 |
| RQ-Kmeans | 6 | 0.0252 | 0.0205 | 0.0364 | 0.0241 |
| RQ-Kmeans | 8 | 0.0279 | 0.0208 | 0.0416 | 0.0255 |
| RQ-Kmeans | 12 | 0.0214 | 0.0179 | 0.0297 | 0.0207 |
| RQ-Kmeans | 16 | 0.0267 | 0.0199 | 0.0400 | 0.0244 |
| OPQ | 2 | 0.0309 | 0.0221 | 0.0485 | 0.0280 |
| OPQ | 3 | 0.0250 | 0.0193 | 0.0424 | 0.0253 |
| OPQ | 4 | 0.0320 | 0.0230 | 0.0525 | 0.0300 |
| OPQ | 6 | **0.0339** | **0.0247** | **0.0556** | **0.0321** |
| OPQ | 8 | 0.0336 | 0.0246 | 0.0447 | 0.0284 |
| OPQ | 12 | 0.0301 | 0.0231 | 0.0433 | 0.0275 |
| OPQ | 16 | 0.0242 | 0.0188 | 0.0354 | 0.0226 |

### Token 利用率 / Codebook Collapse 分析

字段说明：

- `Entropy`：该层 code token 使用频率分布的信息熵，越高说明分布越均匀。
- `Effective usage`：基于熵得到的有效 token 使用比例，越低说明有效容量越小。
- `Gini`：该层 code token 使用频率分布的 Gini 系数，越高说明分布越集中。

| Method | L | Layer | Entropy | Effective usage | Gini |
| --- | ---: | ---: | ---: | ---: | ---: |
| RQ-VAE | 2 | 1 | 5.425 | 88.6% | 0.236 |
| RQ-VAE | 2 | 2 | 4.625 | 40.6% | 0.651 |
| RQ-VAE | 3 | 1 | 5.325 | 80.2% | 0.318 |
| RQ-VAE | 3 | 2 | 4.521 | 36.5% | 0.680 |
| RQ-VAE | 3 | 3 | 4.039 | 22.2% | 0.817 |
| RQ-VAE | 4 | 1 | 5.066 | 61.9% | 0.473 |
| RQ-VAE | 4 | 2 | 4.191 | 26.2% | 0.784 |
| RQ-VAE | 4 | 3 | 3.759 | 17.2% | 0.863 |
| RQ-VAE | 4 | 4 | 2.518 | 5.0% | 0.958 |
| RQ-VAE | 6 | 1 | 4.929 | 54.2% | 0.532 |
| RQ-VAE | 6 | 2 | 4.190 | 25.8% | 0.790 |
| RQ-VAE | 6 | 3 | 2.945 | 7.5% | 0.942 |
| RQ-VAE | 6 | 4 | 2.060 | 3.2% | 0.974 |
| RQ-VAE | 6 | 5 | 1.604 | 2.3% | 0.981 |
| RQ-VAE | 6 | 6 | 0.334 | 0.7% | 0.994 |
| RQ-VAE | 8 | 1 | 4.924 | 54.1% | 0.540 |
| RQ-VAE | 8 | 2 | 3.838 | 18.7% | 0.856 |
| RQ-VAE | 8 | 3 | 2.890 | 7.0% | 0.945 |
| RQ-VAE | 8 | 4 | 1.975 | 2.9% | 0.978 |
| RQ-VAE | 8 | 5 | 1.874 | 2.9% | 0.976 |
| RQ-VAE | 8 | 6 | 0.006 | 0.6% | 0.994 |
| RQ-VAE | 8 | 7 | 0.174 | 0.5% | 0.995 |
| RQ-VAE | 8 | 8 | 0.001 | 0.4% | 0.996 |
| RQ-VAE | 12 | 1 | 4.410 | 32.8% | 0.722 |
| RQ-VAE | 12 | 2 | 1.992 | 2.9% | 0.978 |
| RQ-VAE | 12 | 3 | 2.158 | 3.5% | 0.972 |
| RQ-VAE | 12 | 4 | 1.379 | 1.7% | 0.986 |
| RQ-VAE | 12 | 5 | 0.161 | 0.5% | 0.996 |
| RQ-VAE | 12 | 6 | 0.011 | 0.4% | 0.996 |
| RQ-VAE | 12 | 7 | 0.003 | 0.4% | 0.996 |
| RQ-VAE | 12 | 8 | 0.391 | 0.6% | 0.994 |
| RQ-VAE | 12 | 9 | -0.000 | 0.0% | 0.992 |
| RQ-VAE | 12 | 10 | -0.000 | 0.0% | 0.993 |
| RQ-VAE | 12 | 11 | 0.013 | 0.5% | 0.995 |
| RQ-VAE | 12 | 12 | -0.000 | 0.0% | 0.996 |
| RQ-VAE | 16 | 1 | 0.770 | 0.9% | 0.993 |
| RQ-VAE | 16 | 2 | 0.419 | 1.2% | 0.990 |
| RQ-VAE | 16 | 3 | 0.688 | 1.0% | 0.992 |
| RQ-VAE | 16 | 4 | 0.219 | 0.6% | 0.995 |
| RQ-VAE | 16 | 5 | -0.000 | 0.0% | 0.991 |
| RQ-VAE | 16 | 6 | 0.361 | 0.6% | 0.995 |
| RQ-VAE | 16 | 7 | 1.048 | 2.7% | 0.979 |
| RQ-VAE | 16 | 8 | 0.676 | 0.8% | 0.993 |
| RQ-VAE | 16 | 9 | 0.358 | 0.6% | 0.995 |
| RQ-VAE | 16 | 10 | 1.155 | 1.2% | 0.990 |
| RQ-VAE | 16 | 11 | 0.076 | 0.6% | 0.994 |
| RQ-VAE | 16 | 12 | 0.208 | 0.5% | 0.996 |
| RQ-VAE | 16 | 13 | 0.518 | 0.7% | 0.994 |
| RQ-VAE | 16 | 14 | 0.200 | 0.5% | 0.995 |
| RQ-VAE | 16 | 15 | 0.949 | 1.2% | 0.990 |
| RQ-VAE | 16 | 16 | 0.655 | 1.3% | 0.988 |
| RQ-Kmeans | 2 | 1 | 5.269 | 75.9% | 0.400 |
| RQ-Kmeans | 2 | 2 | 5.297 | 78.0% | 0.387 |
| RQ-Kmeans | 3 | 1 | 5.269 | 75.9% | 0.400 |
| RQ-Kmeans | 3 | 2 | 5.297 | 78.0% | 0.387 |
| RQ-Kmeans | 3 | 3 | 5.232 | 73.1% | 0.431 |
| RQ-Kmeans | 4 | 1 | 5.269 | 75.9% | 0.400 |
| RQ-Kmeans | 4 | 2 | 5.297 | 78.0% | 0.387 |
| RQ-Kmeans | 4 | 3 | 5.232 | 73.1% | 0.431 |
| RQ-Kmeans | 4 | 4 | 5.207 | 71.3% | 0.447 |
| RQ-Kmeans | 6 | 1 | 5.269 | 75.9% | 0.400 |
| RQ-Kmeans | 6 | 2 | 5.297 | 78.0% | 0.387 |
| RQ-Kmeans | 6 | 3 | 5.232 | 73.1% | 0.431 |
| RQ-Kmeans | 6 | 4 | 5.207 | 71.3% | 0.447 |
| RQ-Kmeans | 6 | 5 | 5.181 | 69.5% | 0.460 |
| RQ-Kmeans | 6 | 6 | 5.147 | 67.2% | 0.481 |
| RQ-Kmeans | 8 | 1 | 5.269 | 75.9% | 0.400 |
| RQ-Kmeans | 8 | 2 | 5.297 | 78.0% | 0.387 |
| RQ-Kmeans | 8 | 3 | 5.232 | 73.1% | 0.431 |
| RQ-Kmeans | 8 | 4 | 5.207 | 71.3% | 0.447 |
| RQ-Kmeans | 8 | 5 | 5.181 | 69.5% | 0.460 |
| RQ-Kmeans | 8 | 6 | 5.147 | 67.2% | 0.481 |
| RQ-Kmeans | 8 | 7 | 5.088 | 63.3% | 0.513 |
| RQ-Kmeans | 8 | 8 | 5.040 | 60.4% | 0.535 |
| RQ-Kmeans | 12 | 1 | 5.269 | 75.9% | 0.400 |
| RQ-Kmeans | 12 | 2 | 5.297 | 78.0% | 0.387 |
| RQ-Kmeans | 12 | 3 | 5.232 | 73.1% | 0.431 |
| RQ-Kmeans | 12 | 4 | 5.207 | 71.3% | 0.447 |
| RQ-Kmeans | 12 | 5 | 5.181 | 69.5% | 0.460 |
| RQ-Kmeans | 12 | 6 | 5.147 | 67.2% | 0.481 |
| RQ-Kmeans | 12 | 7 | 5.088 | 63.3% | 0.513 |
| RQ-Kmeans | 12 | 8 | 5.040 | 60.4% | 0.535 |
| RQ-Kmeans | 12 | 9 | 4.989 | 57.3% | 0.564 |
| RQ-Kmeans | 12 | 10 | 4.939 | 54.5% | 0.588 |
| RQ-Kmeans | 12 | 11 | 4.829 | 48.9% | 0.628 |
| RQ-Kmeans | 12 | 12 | 4.881 | 51.5% | 0.614 |
| RQ-Kmeans | 16 | 1 | 5.269 | 75.9% | 0.400 |
| RQ-Kmeans | 16 | 2 | 5.297 | 78.0% | 0.387 |
| RQ-Kmeans | 16 | 3 | 5.232 | 73.1% | 0.431 |
| RQ-Kmeans | 16 | 4 | 5.207 | 71.3% | 0.447 |
| RQ-Kmeans | 16 | 5 | 5.181 | 69.5% | 0.460 |
| RQ-Kmeans | 16 | 6 | 5.147 | 67.2% | 0.481 |
| RQ-Kmeans | 16 | 7 | 5.088 | 63.3% | 0.513 |
| RQ-Kmeans | 16 | 8 | 5.040 | 60.4% | 0.535 |
| RQ-Kmeans | 16 | 9 | 4.989 | 57.3% | 0.564 |
| RQ-Kmeans | 16 | 10 | 4.939 | 54.5% | 0.588 |
| RQ-Kmeans | 16 | 11 | 4.829 | 48.9% | 0.628 |
| RQ-Kmeans | 16 | 12 | 4.881 | 51.5% | 0.614 |
| RQ-Kmeans | 16 | 13 | 4.848 | 49.8% | 0.623 |
| RQ-Kmeans | 16 | 14 | 4.787 | 46.9% | 0.647 |
| RQ-Kmeans | 16 | 15 | 4.727 | 44.1% | 0.667 |
| RQ-Kmeans | 16 | 16 | 4.681 | 42.1% | 0.683 |
| OPQ | 2 | 1 | 5.147 | 67.1% | 0.463 |
| OPQ | 2 | 2 | 5.255 | 74.8% | 0.409 |
| OPQ | 3 | 1 | 5.146 | 67.1% | 0.474 |
| OPQ | 3 | 2 | 5.210 | 71.5% | 0.430 |
| OPQ | 3 | 3 | 5.273 | 76.1% | 0.402 |
| OPQ | 4 | 1 | 5.255 | 74.8% | 0.416 |
| OPQ | 4 | 2 | 5.263 | 75.4% | 0.402 |
| OPQ | 4 | 3 | 5.204 | 71.1% | 0.444 |
| OPQ | 4 | 4 | 5.328 | 80.5% | 0.356 |
| OPQ | 6 | 1 | 5.198 | 70.7% | 0.451 |
| OPQ | 6 | 2 | 5.275 | 76.3% | 0.376 |
| OPQ | 6 | 3 | 5.353 | 82.5% | 0.333 |
| OPQ | 6 | 4 | 5.302 | 78.4% | 0.374 |
| OPQ | 6 | 5 | 5.257 | 74.9% | 0.411 |
| OPQ | 6 | 6 | 5.331 | 80.7% | 0.353 |
| OPQ | 8 | 1 | 5.305 | 78.6% | 0.383 |
| OPQ | 8 | 2 | 5.339 | 81.3% | 0.341 |
| OPQ | 8 | 3 | 5.400 | 86.5% | 0.290 |
| OPQ | 8 | 4 | 5.315 | 79.4% | 0.370 |
| OPQ | 8 | 5 | 5.345 | 81.9% | 0.344 |
| OPQ | 8 | 6 | 5.266 | 75.7% | 0.403 |
| OPQ | 8 | 7 | 5.344 | 81.8% | 0.339 |
| OPQ | 8 | 8 | 5.371 | 84.0% | 0.319 |
| OPQ | 12 | 1 | 5.297 | 78.0% | 0.388 |
| OPQ | 12 | 2 | 5.462 | 92.0% | 0.220 |
| OPQ | 12 | 3 | 5.406 | 87.0% | 0.283 |
| OPQ | 12 | 4 | 5.372 | 84.1% | 0.325 |
| OPQ | 12 | 5 | 5.379 | 84.7% | 0.308 |
| OPQ | 12 | 6 | 5.418 | 88.1% | 0.273 |
| OPQ | 12 | 7 | 5.397 | 86.2% | 0.292 |
| OPQ | 12 | 8 | 5.342 | 81.6% | 0.348 |
| OPQ | 12 | 9 | 5.372 | 84.1% | 0.326 |
| OPQ | 12 | 10 | 5.368 | 83.8% | 0.323 |
| OPQ | 12 | 11 | 5.350 | 82.2% | 0.328 |
| OPQ | 12 | 12 | 5.442 | 90.2% | 0.242 |
| OPQ | 16 | 1 | 5.343 | 81.7% | 0.352 |
| OPQ | 16 | 2 | 5.445 | 90.5% | 0.248 |
| OPQ | 16 | 3 | 5.420 | 88.2% | 0.266 |
| OPQ | 16 | 4 | 5.403 | 86.7% | 0.290 |
| OPQ | 16 | 5 | 5.441 | 90.1% | 0.252 |
| OPQ | 16 | 6 | 5.430 | 89.2% | 0.264 |
| OPQ | 16 | 7 | 5.414 | 87.7% | 0.278 |
| OPQ | 16 | 8 | 5.400 | 86.5% | 0.294 |
| OPQ | 16 | 9 | 5.381 | 84.8% | 0.303 |
| OPQ | 16 | 10 | 5.431 | 89.2% | 0.265 |
| OPQ | 16 | 11 | 5.349 | 82.2% | 0.343 |
| OPQ | 16 | 12 | 5.379 | 84.7% | 0.315 |
| OPQ | 16 | 13 | 5.427 | 88.9% | 0.267 |
| OPQ | 16 | 14 | 5.442 | 90.2% | 0.251 |
| OPQ | 16 | 15 | 5.419 | 88.1% | 0.276 |
| OPQ | 16 | 16 | 5.442 | 90.2% | 0.240 |

直接结论：

- `RQ-VAE` 随 length 增加，后层 `Entropy` 和 `Effective usage` 明显下降，同时 `Gini` 明显升高，说明长码设置下后层有效容量快速收缩。
- `RQ-Kmeans` 各层 `Entropy` 仍保持在较高区间，但随着 layer 加深，`Effective usage` 逐步下降、`Gini` 逐步升高，说明后层分布逐渐集中。
- `OPQ` 各 length、各 layer 的 `Effective usage` 基本维持在较高水平，`Gini` 相对较低，是三者中 token 使用分布最稳定的一类。

### 逐层信息增益：Prefix Projection

下表记录每个长度下 prefix projection 的最佳 `k`。

| Method | L | Best k | Prefix R@10 | Prefix N@10 | Drop vs full R@10 | Mean bucket size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RQ-VAE | 2 | 1 | 0.0119 | 0.0123 | 0.0357 | 115.1 |
| RQ-VAE | 3 | 3 | 0.0328 | 0.0214 | 0.0126 | 1.5 |
| RQ-VAE | 4 | 4 | 0.0320 | 0.0195 | 0.0073 | 1.2 |
| RQ-VAE | 6 | 5 | 0.0237 | 0.0149 | 0.0203 | 1.6 |
| RQ-VAE | 8 | 7 | 0.0436 | 0.0268 | 0.0006 | 1.0 |
| RQ-VAE | 12 | 4 | 0.0202 | 0.0143 | 0.0309 | 6.1 |
| RQ-VAE | 16 | 6 | 0.0171 | 0.0121 | 0.0253 | 9383.4 |
| RQ-Kmeans | 2 | 1 | 0.0146 | 0.0107 | 0.0302 | 171.3 |
| RQ-Kmeans | 3 | 3 | 0.0351 | 0.0236 | 0.0092 | 1.7 |
| RQ-Kmeans | 4 | 4 | 0.0339 | 0.0217 | 0.0030 | 1.2 |
| RQ-Kmeans | 6 | 6 | 0.0364 | 0.0241 | 0.0000 | 1.0 |
| RQ-Kmeans | 8 | 6 | 0.0416 | 0.0255 | 0.0000 | 1.0 |
| RQ-Kmeans | 12 | 6 | 0.0297 | 0.0207 | 0.0000 | 1.0 |
| RQ-Kmeans | 16 | 6 | 0.0400 | 0.0244 | 0.0000 | 1.0 |
| OPQ | 2 | 1 | 0.0304 | 0.0192 | 0.0181 | 319.0 |
| OPQ | 3 | 1 | 0.0231 | 0.0182 | 0.0193 | 271.2 |
| OPQ | 4 | 4 | 0.0221 | 0.0181 | 0.0304 | 4.8 |
| OPQ | 6 | 6 | 0.0487 | 0.0290 | 0.0069 | 1.3 |
| OPQ | 8 | 5 | 0.0447 | 0.0283 | 0.0000 | 1.1 |
| OPQ | 12 | 6 | 0.0433 | 0.0275 | 0.0000 | 1.0 |
| OPQ | 16 | 5 | 0.0354 | 0.0226 | 0.0000 | 1.0 |

主要结论：

- `RQ-Kmeans` 大多在 `k=3~6` 后饱和；当 `L>=8` 时，前 `6` 个 token 已经足以匹配 full projection，后续 token 几乎不再贡献可测的区分能力。
- `OPQ L=6` 的逐层增益最自然：`R@10` 从 `k=1` 的 `0.0139` 逐步增加到 `k=6` 的 `0.0487`。
- `OPQ L>=8` 往往在 `k=5/6` 已经饱和，继续增加 token 主要增加生成负担。
- `RQ-VAE` 的 prefix 行为不稳定，原因是 collapsed layers 会产生异常大的 bucket 或退化 bucket，因此必须结合 token usage 一起解释。

### Mask-Layer Drop 总结

| Method | 现象 | 解释 |
| --- | --- | --- |
| RQ-VAE | 长码设置下 mask 某些层会产生很大或退化的 bucket。 | collapse 让 layer importance 变得噪声很大。 |
| RQ-Kmeans | `L>=8` 时 mask 很多层几乎不掉点。 | 后续层对候选区分基本冗余。 |
| OPQ | `L=6` 仍存在有明显影响的层；`L>=8` 后很多层 mask drop 接近 0。 | `L=6` 是较好的折中点，更长码长开始冗余。 |

### 总结解释

SID length scaling 不单调，是因为 semantic capacity 和 decoder usability 发生分离。

- `RQ-VAE`：深层 codebook collapse，长码没有稳定增加有效语义容量。
- `RQ-Kmeans`：token 分布没有出现 `RQ-VAE` 那种严重收缩，但边际信息早早饱和，短码如 `L=2/3` 已经很有竞争力。
- `OPQ`：token 利用率健康，`L=6` 是目前观察到的最佳折中；超过 `L=6` 后，额外 token 多数变成冗余，同时增加 decoder 生成难度。

适合正文使用的结论：

> 增加 SID 码长并不会单调提升推荐性能。`RQ-VAE` 受限于明显的 codebook collapse；`RQ-Kmeans` 和 `OPQ` 的 token 分布相对更稳定，但逐层信息增益会逐渐饱和。`OPQ` 在 `L=6` 达到最干净的折中：token 使用仍然均衡，prefix 信息增益也仍然有效。

## Artifacts

RQ4：

- 主表 CSV：`tmp/rq4_sid_geometry_20260424_native/paper_main_table_local_clean.csv`
- Native 结果目录：`tmp/rq4_sid_geometry_20260424_native/`
- 最终 case-study 目录：`tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/`
- 脚本：`scripts/analyze_sid_geometry.py`
- 旧详细记录：`RQ4_SID_Geometry_20260423.md`

Code Length：

- 本地诊断目录：`tmp/inference_diagnostics_20260424/`
- 远端诊断目录：`/data/xqp_data/RecSys26/logs/code_length_study/amazon23_vg/20260421_clean_suffix_mainalign_t5small/inference_diagnostics_20260424`
- 脚本：`scripts/length_inference_diagnostics.py`
- 旧详细记录：`ID_Length_Inference_Diagnostics_20260424.md`
