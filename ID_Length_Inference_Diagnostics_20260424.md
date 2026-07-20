# ID Length Inference-Only Diagnostics

Date: 2026-04-24

Scope: Amazon23-VG clean suffix length study, `t5-small` main-table-aligned decoder exports.

Artifacts:

- Remote output: `/data/xqp_data/RecSys26/logs/code_length_study/amazon23_vg/20260421_clean_suffix_mainalign_t5small/inference_diagnostics_20260424`
- Local copy: `tmp/inference_diagnostics_20260424`
- Script: `scripts/length_inference_diagnostics.py`

## Protocol

Two inference-only diagnostics were run for `RQ-VAE`, `RQ-Kmeans`, and `OPQ` over lengths `L={2,3,4,6,8,12,16}`.

1. Token usage / codebook collapse:
   - Per code position: used-token ratio, dead-token count, entropy, normalized entropy, effective token count, Gini, top-1 mass, top-5 mass.
   - Source: tokenizer `raw_codes.npy` under `tokenizer_suffix/{method}/L{L}`.

2. Layer-wise information gain:
   - Prefix projection: use the first `k` semantic tokens as a bucket key, expand candidates inside the bucket, and evaluate warm-only metrics.
   - Mask projection: remove one layer from the code key, expand candidates sharing all remaining layers, and measure metric drop.
   - Source: existing `pred_topk.jsonl`; no decoder retraining and no beam-search regeneration.
   - Tie-breaker inside a SID bucket: global training-set item popularity, then item id.

Boundary: this second diagnostic measures how much discriminative information is carried by SID prefixes/layers given the existing decoder candidate list. It is not a strict GPU beam-search ablation with `max_new_tokens=k`.

## Baseline Metrics

Warm-only full-test metrics from existing decoder exports:

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

Note: under this exact `metrics_full_warm_only.json` source, `RQ-VAE` has best `R@5` at `L=2`, but best `R@10` at `L=12`. The collapse analysis below explains why long suffixes are unreliable, but the final claim should specify which metric is used.

## Token Usage Findings

| Method | Key observation | Evidence |
| --- | --- | --- |
| RQ-VAE | Severe codebook collapse as length grows. Later layers become almost dead. | Last-layer usage drops from `43.0%` at `L=2` to `7.3%` at `L=4`, `1.9%` at `L=6`, `0.8%` at `L=8`, and `0.4%` at `L=12`. Last-layer normalized entropy drops from `0.837` to near `0`. Gini rises to about `0.996`. |
| RQ-Kmeans | No dead-token collapse, but distribution concentration increases with length. | Usage stays `100%`, but last-layer normalized entropy declines from `0.955` at `L=2` to `0.844` at `L=16`; last-layer Gini rises from `0.387` to `0.683`. |
| OPQ | No codebook collapse; token usage remains stable and relatively balanced. | Usage stays `100%`; normalized entropy is high and improves with length, from about `0.938` mean at `L=2` to `0.976` mean at `L=16`; max Gini decreases from about `0.463` to `0.352`. |

Interpretation:

- `RQ-VAE` length degradation is strongly consistent with codebook collapse. Adding more residual layers does not reliably add semantic capacity because later layers are barely used.
- `RQ-Kmeans` does not collapse in the dead-token sense, but longer codes become increasingly skewed. The extra layers are valid tokens, yet their marginal information is weak.
- `OPQ` is the cleanest tokenizer-side scaling case: longer codes keep using the codebook. Its performance peak at `L=6` is therefore more likely caused by decoder/generation difficulty and marginal-gain saturation, not tokenizer collapse.

## Prefix Information Gain

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

Main reading:

- `RQ-Kmeans` saturates around `k=3` to `k=6`. For `L>=8`, using only the first `6` tokens already matches the full projection, so later tokens add almost no measurable discriminative gain.
- `OPQ L=6` shows the most natural gradual gain: `R@10` rises from `0.0139` at `k=1` to `0.0209`, `0.0299`, `0.0374`, `0.0399`, and `0.0487` at `k=6`. This supports the interpretation that `L=6` is the best balance between sufficient discrimination and manageable generation length.
- `OPQ L>=8` saturates earlier (`k=5` or `k=6`), so extra tokens after that are mostly redundant while still increasing decoder generation length.
- `RQ-VAE` is unstable: some long settings have pathological bucket behavior, consistent with the collapse evidence.

## Mask-Layer Drop

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

Main reading:

- For `RQ-Kmeans` and `OPQ`, once length is long enough, many masked layers cause zero or near-zero drop under this projection diagnostic. This is direct evidence that the later layers are often redundant for candidate discrimination.
- `OPQ L=6` still has a non-trivial damaging layer (`drop=0.0159`), while `OPQ L>=8` has near-zero drop. This aligns with the performance peak at `L=6`.
- `RQ-VAE` mask behavior is noisy because collapsed layers create very large or degenerate buckets; its prefix/mask curves should be interpreted together with token usage, not alone.

## Current Conclusion

The two diagnostics support the following explanation:

1. `RQ-VAE` does not scale cleanly with length because deeper codebooks collapse. Longer codes do not mean more useful semantic capacity.
2. `RQ-Kmeans` uses all tokens, but marginal information saturates early. This explains why short lengths such as `L=2/3` are competitive: longer codes add generation burden but little new discrimination.
3. `OPQ` avoids tokenizer collapse, and `L=6` is the best observed trade-off. Shorter OPQ codes are too coarse; longer codes are mostly redundant after about `k=5/6` and hurt decoder efficiency.

For paper writing, the strongest clean claim is:

> Length scaling is not monotonic because semantic-ID capacity and decoder usability diverge. RQ-VAE suffers from explicit codebook collapse; RQ-Kmeans and OPQ avoid dead-code collapse, but their layer-wise information gain saturates, with OPQ reaching the best balance at `L=6`.

