# Current Experiment Results

Date: 2026-04-25

This document consolidates the current experimental results for:

- Semantic ID length scaling
- Inference-only diagnostics for token usage / codebook collapse and layer-wise information gain
- RQ4 SID geometry alignment using `Jaccard@20` and `RBO@20`, including the final case-study figures

## 1. Artifact Index

| Experiment | Main artifact | Notes |
| --- | --- | --- |
| ID length inference diagnostics | `ID_Length_Inference_Diagnostics_20260424.md` | Detailed token-usage, prefix-projection, and mask-layer tables |
| Local diagnostic outputs | `tmp/inference_diagnostics_20260424/` | Local copy of CSV/JSON/MD outputs |
| Remote diagnostic outputs | `/data/xqp_data/RecSys26/logs/code_length_study/amazon23_vg/20260421_clean_suffix_mainalign_t5small/inference_diagnostics_20260424` | Source output directory on `target-server` |
| RQ4 geometry summary | `RQ4_SID_Geometry_20260423.md` | Protocol and RQ4 decision log |
| Final RQ4 case-study figure | `tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/rq4_case_study_selected.png` | Final styled local-scatter case study |
| Final RQ4 overview figure | `tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/rq4_paper_overview_local.png` | Jaccard/RBO overview figure |

## 2. Semantic ID Length Scaling Results

Dataset and decoder setting:

- Dataset: `amazon23_vg`
- Evaluation: `full` test mode, `warm_only`
- Decoder: existing `t5-small` main-table-aligned clean suffix exports
- Methods: `RQ-VAE`, `RQ-Kmeans`, `OPQ`
- Lengths: `L={2,3,4,6,8,12,16}`

### 2.1 Full Decoder Warm-Only Results

| Method | L | R@5 | N@5 | R@10 | N@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RQ-VAE | 2 | 0.0313 | 0.0219 | 0.0476 | 0.0276 |
| RQ-VAE | 3 | 0.0292 | 0.0212 | 0.0454 | 0.0268 |
| RQ-VAE | 4 | 0.0187 | 0.0157 | 0.0393 | 0.0227 |
| RQ-VAE | 6 | 0.0256 | 0.0187 | 0.0440 | 0.0248 |
| RQ-VAE | 8 | 0.0280 | 0.0214 | 0.0442 | 0.0270 |
| RQ-VAE | 12 | 0.0266 | 0.0194 | 0.0511 | 0.0277 |
| RQ-VAE | 16 | 0.0261 | 0.0188 | 0.0424 | 0.0244 |
| RQ-Kmeans | 2 | 0.0294 | 0.0200 | 0.0448 | 0.0252 |
| RQ-Kmeans | 3 | 0.0270 | 0.0217 | 0.0443 | 0.0276 |
| RQ-Kmeans | 4 | 0.0240 | 0.0188 | 0.0369 | 0.0233 |
| RQ-Kmeans | 6 | 0.0252 | 0.0205 | 0.0364 | 0.0241 |
| RQ-Kmeans | 8 | 0.0279 | 0.0208 | 0.0416 | 0.0255 |
| RQ-Kmeans | 12 | 0.0214 | 0.0179 | 0.0297 | 0.0207 |
| RQ-Kmeans | 16 | 0.0267 | 0.0199 | 0.0400 | 0.0244 |
| OPQ | 2 | 0.0309 | 0.0221 | 0.0485 | 0.0280 |
| OPQ | 3 | 0.0250 | 0.0193 | 0.0424 | 0.0253 |
| OPQ | 4 | 0.0320 | 0.0230 | 0.0525 | 0.0300 |
| OPQ | 6 | 0.0339 | 0.0247 | 0.0556 | 0.0321 |
| OPQ | 8 | 0.0336 | 0.0246 | 0.0447 | 0.0284 |
| OPQ | 12 | 0.0301 | 0.0231 | 0.0433 | 0.0275 |
| OPQ | 16 | 0.0242 | 0.0188 | 0.0354 | 0.0226 |

Important note:

- Under the exact `metrics_full_warm_only.json` source, `RQ-VAE` is best at `L=2` for `R@5`, but best at `L=12` for `R@10`.
- `RQ-Kmeans` is close between `L=2` and `L=3`: `L=2` is slightly better on `R@10`, while `L=3` is better on `N@10`.
- `OPQ L=6` is consistently the best OPQ setting on both `R@10` and `N@10`.

## 3. Inference-Only Diagnostics

These diagnostics were run with:

- Script: `scripts/length_inference_diagnostics.py`
- No decoder retraining
- No beam-search regeneration
- Existing `pred_topk.jsonl` files only

The prefix/mask analysis is a post-hoc projection diagnostic:

- Prefix projection: use the first `k` semantic tokens as a bucket key, expand items in that bucket, and evaluate.
- Mask projection: remove one SID layer from the code key, expand items sharing all remaining layers, and evaluate.
- Tie-breaker inside each SID bucket: global training-set item popularity, then item id.

This measures how much discriminative information is carried by SID prefixes/layers, but it is not a strict model-level beam-search ablation.

### 3.1 Token Usage / Codebook Collapse

| Method | Main observation | Evidence |
| --- | --- | --- |
| RQ-VAE | Severe codebook collapse as length grows. Later layers become almost dead. | Last-layer usage drops from `43.0%` at `L=2` to `7.3%` at `L=4`, `1.9%` at `L=6`, `0.8%` at `L=8`, and `0.4%` at `L=12`. Last-layer normalized entropy approaches `0`, and Gini approaches `1`. |
| RQ-Kmeans | No dead-token collapse, but later layers become increasingly concentrated. | Usage remains `100%`, but last-layer normalized entropy declines from about `0.955` at `L=2` to `0.844` at `L=16`; last-layer Gini rises from about `0.387` to `0.683`. |
| OPQ | No codebook collapse; token usage remains stable and balanced. | Usage remains `100%`; normalized entropy is high and stable, and max Gini decreases from about `0.463` to `0.352` as length grows. |

Interpretation:

- `RQ-VAE` length degradation is strongly consistent with codebook collapse.
- `RQ-Kmeans` uses all tokens, but the marginal information in later layers weakens.
- `OPQ` does not fail at the tokenizer-utilization level; its length optimum is more likely controlled by decoder usability and marginal-gain saturation.

### 3.2 Prefix Information Gain

Best prefix projection per length:

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

Main observations:

- `RQ-Kmeans` saturates around `k=3` to `k=6`. For `L>=8`, the first `6` tokens already recover the full projected list, so later tokens add little measurable discrimination.
- `OPQ L=6` shows the cleanest gradual gain: `R@10` increases from `0.0139` at `k=1` to `0.0487` at `k=6`.
- `OPQ L>=8` usually saturates at `k=5/6`, so tokens beyond that are mostly redundant while increasing generation length.
- `RQ-VAE` has unstable long-code behavior, consistent with its codebook collapse.

### 3.3 Mask-Layer Drop

Most damaging masked layer per length:

| Method | L | Masked layer | R@10 after mask | Drop vs full R@10 | Mean bucket size |
| --- | ---: | ---: | ---: | ---: | ---: |
| RQ-VAE | 2 | 2 | 0.0119 | 0.0357 | 115.1 |
| RQ-VAE | 3 | 3 | 0.0139 | 0.0315 | 5.4 |
| RQ-VAE | 4 | 3 | 0.0165 | 0.0228 | 2.4 |
| RQ-VAE | 6 | 2 | 0.0123 | 0.0317 | 2.5 |
| RQ-VAE | 8 | 1 | 0.0197 | 0.0245 | 3.2 |
| RQ-VAE | 12 | 1 | 0.0091 | 0.0420 | 116.3 |
| RQ-VAE | 16 | 4 | 0.0117 | 0.0307 | 1142.0 |
| RQ-Kmeans | 2 | 2 | 0.0146 | 0.0302 | 171.3 |
| RQ-Kmeans | 3 | 3 | 0.0169 | 0.0274 | 10.7 |
| RQ-Kmeans | 4 | 3 | 0.0273 | 0.0096 | 1.7 |
| RQ-Kmeans | 6 | 6 | 0.0308 | 0.0056 | 1.2 |
| RQ-Kmeans | 8 | 1 | 0.0416 | 0.0000 | 1.0 |
| RQ-Kmeans | 12 | 1 | 0.0297 | 0.0000 | 1.0 |
| RQ-Kmeans | 16 | 1 | 0.0400 | 0.0000 | 1.0 |
| OPQ | 2 | 1 | 0.0161 | 0.0324 | 167.8 |
| OPQ | 3 | 2 | 0.0116 | 0.0308 | 18.7 |
| OPQ | 4 | 2 | 0.0136 | 0.0389 | 7.8 |
| OPQ | 6 | 2 | 0.0397 | 0.0159 | 1.8 |
| OPQ | 8 | 2 | 0.0447 | 0.0000 | 1.0 |
| OPQ | 12 | 1 | 0.0433 | 0.0000 | 1.0 |
| OPQ | 16 | 1 | 0.0354 | 0.0000 | 1.0 |

Interpretation:

- `RQ-Kmeans` and `OPQ` both show layer redundancy after the useful prefix has become sufficiently discriminative.
- `OPQ L=6` still has a meaningful damaging layer, while `OPQ L>=8` has near-zero drops, which supports the claim that `L=6` is the best capacity-usability tradeoff.
- `RQ-VAE` mask behavior is noisy because collapsed layers create degenerate buckets; token usage should be treated as the stronger evidence.

## 4. RQ4: SID Geometry Alignment

RQ4:

> How do different semantic ID designs affect the learning of item semantics?

Final protocol:

- Keep each semantic-ID method in its native/original generation setting.
- Use a unified external semantic reference space: `amazon23_vg.emb-t5-tdcb.npy`.
- Compare the final SID-induced neighborhood with the reference semantic neighborhood.

Main metrics:

- `Jaccard@20`: overlap between the reference top-20 neighbor set and the SID top-20 neighbor set.
- `RBO@20 (p=0.9)`: rank-biased overlap between the two top-20 ranked neighbor lists.

Native distance setting:

| Method | Native SID distance |
| --- | --- |
| I02-TIGER | prefix |
| I03-SEATER | prefix |
| I04-SemID | prefix |
| I05-LETTER | prefix |
| I06-DiffGRM | hamming |
| I07-RQKmeans | prefix |

### 4.1 RQ4 Main Table

Current recorded native-distance main table:

| Method | Native Metric | Jaccard@20 | RBO@20 (p=0.9) |
| --- | --- | ---: | ---: |
| I02-TIGER | prefix | 0.069283 | 0.101570 |
| I03-SEATER | prefix | 0.013739 | 0.018789 |
| I04-SemID | prefix | 0.026286 | 0.021477 |
| I05-LETTER | prefix | 0.038517 | 0.048360 |
| I06-DiffGRM | hamming | 0.172434 | 0.180635 |
| I07-RQKmeans | prefix | 0.128040 | 0.196671 |

Latest styled-figure run table:

| Method | Jaccard@20 | RBO@20 (p=0.9) |
| --- | ---: | ---: |
| I02-TIGER | 0.065565 | 0.096202 |
| I03-SEATER | 0.011968 | 0.015047 |
| I04-SemID | 0.024311 | 0.026223 |
| I05-LETTER | 0.033922 | 0.041583 |
| I06-DiffGRM | 0.193558 | 0.178106 |
| I07-RQKmeans | 0.138946 | 0.194397 |

The two tables are numerically close but not identical because the final styled figure was regenerated in a later plotting/output directory. For paper-facing numbers, use one table consistently and cite the exact output directory.

### 4.2 RQ4 Interpretation

Main reading:

- `I06-DiffGRM` has the strongest Jaccard-style neighborhood-member preservation.
- `I07-RQKmeans` has the strongest RBO-style front-rank preservation.
- `I02-TIGER` improves under native-prefix distance compared with the older all-Hamming protocol, but it is still below `I06/I07`.
- `I03-SEATER` remains the weakest under the current final SID geometry metric.
- `I04-SemID` and `I05-LETTER` are intermediate but closer to the lower-performing group than to `I06/I07`.

Paper-safe wording:

> `Jaccard@20` and `RBO@20` show that semantic-ID designs differ substantially in how well their final discrete geometry preserves item semantics. `DiffGRM` better preserves local neighborhood membership, while `RQ-Kmeans` better preserves front-ranked neighbor order. This suggests that different SID constructions encode semantic structure in measurably different ways, beyond downstream decoder performance alone.

## 5. RQ4 Figures

### 5.1 Jaccard/RBO Overview Figure

Path:

`tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/rq4_paper_overview_local.png`

![RQ4 Jaccard/RBO overview](tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/rq4_paper_overview_local.png)

### 5.2 Final Case Study Figure

Path:

`tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/rq4_case_study_selected.png`

![RQ4 case study](tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/rq4_case_study_selected.png)

Case-study setting:

- Anchor item: `18089`
- Marker semantics:
  - Blue circle: SID top-K
  - Red triangle: reference top-K
  - Purple diamond: overlap
  - Gray triangle: local context
  - Black star: anchor item
- Style:
  - Low-saturation DECOR-like palette
  - No reference rings
  - No anchor-to-SID lines
  - Compact `2x3` panel layout

Per-case metrics embedded in the figure:

| Method | Jaccard@20 | RBO@20 |
| --- | ---: | ---: |
| I02-TIGER | 0.08 | 0.14 |
| I03-SEATER | 0.00 | 0.00 |
| I04-SemID | 0.00 | 0.00 |
| I05-LETTER | 0.05 | 0.02 |
| I06-DiffGRM | 0.48 | 0.28 |
| I07-RQKmeans | 0.33 | 0.38 |

This case is useful because:

- It shows visible separation across SID designs.
- `I06/I07` are clearly stronger, but the figure is not as extreme as earlier rejected cases.
- `I02/I05` retain some middle-layer signal, while `I03/I04` remain weak.
- The plot uses real local scatter rather than a schematic radial layout.

## 6. Current Consolidated Conclusion

The current results support two connected claims.

First, semantic-ID length scaling is non-monotonic because additional code length does not always translate into useful item discrimination:

- `RQ-VAE` fails mainly because long residual codebooks collapse.
- `RQ-Kmeans` avoids dead-code collapse, but its layer-wise information gain saturates early.
- `OPQ` keeps healthy code usage, and `L=6` appears to be the best tradeoff between enough semantic discrimination and manageable decoder generation length.

Second, different SID designs preserve item semantics differently at the representation level:

- `DiffGRM` and `RQ-Kmeans` best preserve local semantic neighborhoods under `Jaccard@20/RBO@20`.
- `TIGER`, `LETTER`, `SemID`, and `SEATER` do not recover the external semantic neighborhood as strongly, even if some of them remain useful for generative decoding.
- This supports the RQ4 framing that SID design affects how item semantics are represented and learned, not only downstream recommendation accuracy.

## 7. Open Follow-Up

If we want a stricter version of the layer-wise information-gain experiment, the next step is GPU-based model-level ablation:

- Prefix-only generation: rerun beam search with `max_new_tokens=k` and map prefix buckets to item candidates.
- Layer-mask generation: modify decoder inputs or SID constraints to mask a specific target layer during generation.

The current inference-only projection is sufficient for fast diagnosis, but the stricter beam-search version would be better if this result becomes a major paper claim.

