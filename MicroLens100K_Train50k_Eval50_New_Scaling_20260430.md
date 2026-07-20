# MicroLens100K Train50K/Eval50 新 Scaling 实验结果

生成时间：2026-04-30  
远端根目录：`/data/xqp_data/RecSys26`  
实验目录：`logs/scaling_microlens100k_train50k_eval_20k`

## 1. 实验口径

本文件只记录已经从远端日志和 `metrics*.json` 文件确认过的新 scaling 结果。

- 数据划分：`microlens_100k_train_50k_eval`
- 固定评测集：`third_party/SETRec/data/microlens_100k_train_50k_eval/eval50_testing_dict.npy`
- warm item 集合：`third_party/SETRec/data/microlens_100k_train_50k_eval/eval50_warm_item.npy`
- 评测口径：`warm-only`
- 导出口径：`top100`
- decoder 训练：`iterations=20000`, `save_model_every=5000`, `early_stop_patience=3`, `num_user_tokens=1`
- 主表指标：`full warm-only`
- 辅助指标：`loo warm-only`

需要注意，`I05-letter` 的 `top100 strict` 会因为部分用户候选不足 100 而失败，因此当前 LETTER 结果按用户已确认的 `top100 non-strict` 口径报告。`I07-rkmeans` 当前 strict 口径下 `short_users_after_candidate_filter=0`。

## 2. 主结果表：full warm-only

| ID | Model | 状态 | Run | Eval strict | Short users | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| I05-letter | M01-t5s | completed | `I05-letter/M01-t5s/20260429_064517__gpu1` | false | 337 | 0.0168 | 0.0113 | 0.0267 | 0.0148 |
| I05-letter | M02-t5b | completed | `I05-letter/M02-t5b/20260429_121210__gpu1` | false | 441 | 0.0219 | 0.0137 | 0.0381 | 0.0191 |
| I05-letter | M03-t5l | running | `I05-letter/M03-t5l/20260429_174221__gpu5` | pending | pending | pending | pending | pending | pending |
| I07-rkmeans | M01-t5s | completed | `I07-rkmeans/M01-t5s/20260428_125537__gpu3` | true | 0 | 0.0177 | 0.0117 | 0.0262 | 0.0146 |
| I07-rkmeans | M02-t5b | completed | `I07-rkmeans/M02-t5b/20260428_225316__gpu3` | true | 0 | 0.0215 | 0.0153 | 0.0352 | 0.0199 |
| I07-rkmeans | M03-t5l | completed | `I07-rkmeans/M03-t5l/20260428_225316__gpu6` | true | 0 | 0.0141 | 0.0090 | 0.0218 | 0.0116 |

当前可确认现象：

- `I05-letter`: small 到 base 明显提升，full `R@10` 从 `0.0267` 提升到 `0.0381`。
- `I07-rkmeans`: base 最好，large final 明显掉点，full `R@10` 从 base 的 `0.0352` 降到 large final 的 `0.0218`。
- `I05-letter-large` 还没有完成，不能参与 scaling 结论。

## 3. 辅助结果表：loo warm-only

| ID | Model | 状态 | Run | Eval strict | Short users | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| I05-letter | M01-t5s | completed | `I05-letter/M01-t5s/20260429_064517__gpu1` | false | 173 | 0.0178 | 0.0115 | 0.0279 | 0.0148 |
| I05-letter | M02-t5b | completed | `I05-letter/M02-t5b/20260429_121210__gpu1` | false | 245 | 0.0201 | 0.0118 | 0.0330 | 0.0160 |
| I05-letter | M03-t5l | running | `I05-letter/M03-t5l/20260429_174221__gpu5` | pending | pending | pending | pending | pending | pending |
| I07-rkmeans | M01-t5s | completed | `I07-rkmeans/M01-t5s/20260428_125537__gpu3` | true | 0 | 0.0178 | 0.0113 | 0.0254 | 0.0137 |
| I07-rkmeans | M02-t5b | completed | `I07-rkmeans/M02-t5b/20260428_225316__gpu3` | true | 0 | 0.0198 | 0.0130 | 0.0332 | 0.0173 |
| I07-rkmeans | M03-t5l | completed | `I07-rkmeans/M03-t5l/20260428_225316__gpu6` | true | 0 | 0.0148 | 0.0094 | 0.0224 | 0.0118 |

## 4. RQ-Kmeans Large Checkpoint 诊断

针对 `I07-rkmeans/M03-t5l/20260428_225316__gpu6`，已对部分 checkpoint 做过 top100 warm-only 导出与评测。

| Checkpoint | Mode | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---:|---|---:|---:|---:|---:|
| 4999 | full | 0.0208 | 0.0151 | 0.0330 | 0.0192 |
| 9999 | full | 0.0186 | 0.0126 | 0.0287 | 0.0159 |
| 19999 | full | 0.0141 | 0.0090 | 0.0218 | 0.0116 |
| 4999 | loo | 0.0201 | 0.0134 | 0.0309 | 0.0168 |
| 9999 | loo | 0.0205 | 0.0136 | 0.0293 | 0.0164 |
| 19999 | loo | 0.0148 | 0.0094 | 0.0224 | 0.0118 |

诊断结论：

- large final 的低值不是由 strict eval 缺预测或候选不足造成的，`prediction_users=6482`，`short_users_after_candidate_filter=0`。
- checkpoint sweep 显示 `ckpt_4999 > ckpt_9999 > ckpt_19999`，说明 large 后期训练出现明显退化。
- 当前证据更支持“large decoder 后期过拟合或优化/正则不适配”，而不是 ID 文件、导出脚本或 warm-only 评测脚本错误。

## 5. LETTER Large 当前状态

当前有效运行：

- Run：`logs/scaling_microlens100k_train50k_eval_20k/I05-letter/M03-t5l/20260429_174221__gpu5`
- 训练进程：`4094176`
- GPU：`gpu5`
- 启动方式：原训练口径，额外设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 最新确认进度：约 `12660 / 20000` step
- 最新确认 loss：约 `0.3183`
- 最新确认 GPU5 显存：约 `48412 / 49140 MiB`
- 状态判断：已通过前几步 forward，没有立刻 OOM，但显存非常贴边

### 5.1 LETTER Large 中间 checkpoint 评测

对 `I05-letter/M03-t5l/20260429_174221__gpu5` 的 `checkpoint_4999.pt` 和 `checkpoint_9999.pt` 已在 GPU1 上完成 top100 warm-only non-strict 导出评测。

| Checkpoint | Mode | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Short users |
|---:|---|---:|---:|---:|---:|---:|
| 4999 | full | 0.0206 | 0.0134 | 0.0357 | 0.0186 | 496 |
| 9999 | full | 0.0148 | 0.0098 | 0.0245 | 0.0131 | 631 |
| 4999 | loo | 0.0205 | 0.0128 | 0.0341 | 0.0172 | 271 |
| 9999 | loo | 0.0157 | 0.0097 | 0.0247 | 0.0126 | 339 |

中间结论：

- `checkpoint_4999` 明显优于 `checkpoint_9999`，说明 `I05-letter-large` 也出现了与 `I07-rkmeans-large` 类似的后期退化趋势。
- `checkpoint_4999` 的 full `R@10=0.0357`，接近 `I05-letter-base` 的 full `R@10=0.0381`，但仍略低。
- `checkpoint_9999` 已经明显低于 base，继续等 final 结果时需要重点判断是否进一步退化。

此前失败的 large 尝试：

| Run | GPU | 状态 | 原因 |
|---|---:|---|---|
| `I05-letter/M03-t5l/20260429_165148__gpu5` | 5 | failed | OOM |
| `I05-letter/M03-t5l/20260429_170532__gpu3` | 3 | failed | OOM |
| `I05-letter/M03-t5l/20260429_173012__gpu6` | 6 | failed | OOM |

## 6. 待完成事项

- 等待 `I05-letter/M03-t5l/20260429_174221__gpu5` 完成训练、导出和评测。
- LETTER large 完成后，按 `top100 non-strict` 补充 `full warm-only` 与 `loo warm-only` 结果。
- 如果 LETTER large 再次 OOM，需要优先等待更空的 48G GPU；若改 batch 或 gradient 设置，需要单独标记为改口径实验，不能与当前表格直接混用。
- RQ-Kmeans large 的 `ckpt_14999` sweep 之前被停止，当前没有落盘 metrics；如果需要更完整的退化曲线，可以后续单独补跑。

## 7. 结果来源

主结果 metrics 文件：

- `logs/scaling_microlens100k_train50k_eval_20k/I05-letter/M01-t5s/20260429_064517__gpu1/export/metrics_full_warm_only_eval50_candidate_top100_nonstrict.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I05-letter/M02-t5b/20260429_121210__gpu1/export/metrics_full_warm_only_eval50_candidate_top100_nonstrict.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M01-t5s/20260428_125537__gpu3/export/metrics_full_warm_only_eval50_candidate.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M02-t5b/20260428_225316__gpu3/export/metrics_full_warm_only_eval50_candidate.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M03-t5l/20260428_225316__gpu6/export/metrics_full_warm_only_eval50_candidate.json`

辅助结果 metrics 文件：

- `logs/scaling_microlens100k_train50k_eval_20k/I05-letter/M01-t5s/20260429_064517__gpu1/export/metrics_loo_warm_only_eval50_candidate_top100_nonstrict.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I05-letter/M02-t5b/20260429_121210__gpu1/export/metrics_loo_warm_only_eval50_candidate_top100_nonstrict.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M01-t5s/20260428_125537__gpu3/export/metrics_loo_warm_only_eval50_candidate.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M02-t5b/20260428_225316__gpu3/export/metrics_loo_warm_only_eval50_candidate.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M03-t5l/20260428_225316__gpu6/export/metrics_loo_warm_only_eval50_candidate.json`

checkpoint sweep metrics 文件：

- `logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M03-t5l/20260428_225316__gpu6/checkpoint_sweep_top100/ckpt_4999/metrics_full_warm_only_eval50_candidate.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M03-t5l/20260428_225316__gpu6/checkpoint_sweep_top100/ckpt_9999/metrics_full_warm_only_eval50_candidate.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M03-t5l/20260428_225316__gpu6/checkpoint_sweep_top100/ckpt_4999/metrics_loo_warm_only_eval50_candidate.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M03-t5l/20260428_225316__gpu6/checkpoint_sweep_top100/ckpt_9999/metrics_loo_warm_only_eval50_candidate.json`

LETTER large 中间 checkpoint metrics 文件：

- `logs/scaling_microlens100k_train50k_eval_20k/I05-letter/M03-t5l/20260429_174221__gpu5/checkpoint_eval_top100/ckpt_4999/metrics_full_warm_only_eval50_candidate_top100_nonstrict.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I05-letter/M03-t5l/20260429_174221__gpu5/checkpoint_eval_top100/ckpt_9999/metrics_full_warm_only_eval50_candidate_top100_nonstrict.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I05-letter/M03-t5l/20260429_174221__gpu5/checkpoint_eval_top100/ckpt_4999/metrics_loo_warm_only_eval50_candidate_top100_nonstrict.json`
- `logs/scaling_microlens100k_train50k_eval_20k/I05-letter/M03-t5l/20260429_174221__gpu5/checkpoint_eval_top100/ckpt_9999/metrics_loo_warm_only_eval50_candidate_top100_nonstrict.json`
