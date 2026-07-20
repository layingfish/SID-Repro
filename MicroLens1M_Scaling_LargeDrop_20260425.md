# MicroLens-1M Scaling Large-Drop Study

## Goal

Check whether the large-model performance drop observed on MicroLens-50K is mitigated by using the larger MicroLens-1M dataset.

Target IDs:

| ID | Method | Reason |
|---|---|---|
| I04 | SemID | Large-model drop observed in previous MicroLens-50K scaling runs |
| I05 | LETTER | Large-model drop observed in previous MicroLens-50K scaling runs |
| I07 | RQ-Kmeans | Large-model drop observed in previous MicroLens-50K scaling runs |

Model sizes:

| Size | Backbone |
|---|---|
| small | T5-small |
| base | T5-base |
| large | T5-large |

Primary evaluation:

| Protocol | Notes |
|---|---|
| warm-only | Same reporting style as current scaling tables |
| LOO and full | Keep both metrics when export finishes |

## Dataset

Official source: `https://recsys.westlake.edu.cn/MicroLens-1M-Dataset/`

Downloaded on `target-server` only:

| File | Remote path | Size |
|---|---|---:|
| interactions | `/data/xqp_data/RecSys26/datasets/microlens_1m/raw/MicroLens-1M_pairs.csv` | 241M |
| titles | `/data/xqp_data/RecSys26/datasets/microlens_1m/raw/MicroLens-1M_title.csv` | 8.9M |
| item map | `/data/xqp_data/RecSys26/datasets/microlens_1m/raw/MicroLens-1M_items.tsv` | 1.2M |

Important parsing note:

`MicroLens-1M_title.csv` has no header and contains unbalanced quotes in title text. It must be parsed by splitting each physical line at the first comma, not by standard CSV logical-row parsing.

## Preprocessing Status

| Step | Status | Output |
|---|---|---|
| raw download | done | `/data/xqp_data/RecSys26/datasets/microlens_1m/raw/` |
| SETRec split | done | `/data/xqp_data/RecSys26/third_party/SETRec/data/microlens_1m/` |
| item text map | done | `combine_tdcb_maps.npy` |
| T5 item embeddings | done | `microlens_1m.emb-t5-tdcb.npy`, shape `(88866, 768)` |
| RQ_VAE_Recommender raw split | done | `/data/xqp_data/RecSys26/datasets/rqvae_recommender/setrec/raw/microlens_1m/` |
| strict raw split | done | `/data/xqp_data/RecSys26/datasets/rqvae_recommender/setrec/raw/microlens_1m_strict/` |
| RQ_VAE_Recommender processed split | done | `/data/xqp_data/RecSys26/datasets/rqvae_recommender/setrec/processed/data_microlens_1m.pt` |

SETRec split summary:

| Field | Value |
|---|---:|
| raw interactions | 9,095,620 |
| raw users | 1,000,000 |
| after 5-core users | 998,842 |
| after min-train-len users | 922,212 |
| final items | 88,866 |
| k-core | 5 |

## Current Implementation Changes

| File | Change |
|---|---|
| `scripts/build_setrec_combine_tdcb_maps.py` | Add `microlens_1m`; line-based parser for no-header MicroLens-1M title file |
| `scripts/build_seater_setrec_tsv.py` | Add `microlens_1m` domain choice |
| `scripts/unified_eval.py` | Add `microlens_1m` dataset choice |
| `scripts/scaling/run_experiment.py` | Allow GPUs `0,1,3,4,5,6` instead of only `0,1,3,4` |
| `scripts/scaling/build_microlens_semid_paths.py` | Build MicroLens title-derived SemID paths as `cluster/topic/sub/leaf` |
| `scripts/scaling/launch_microlens1m_large_drop_cell.sh` | Common train/export/eval launcher for this MicroLens-1M large-drop study |

## Tokenizer Status

| ID | Status | Output | Notes |
|---|---|---|---|
| I04-SemID | ready | `logs/scaling_microlens1m_large_drop/I04-semid/tiger_compat_title_hierarchy/` | Uses title-derived hierarchy, not the invalid one-token category fallback |
| I05-LETTER-compatible | ready | `logs/scaling_microlens1m_large_drop/I05-letter/tiger_compat/` | Matches the previous MicroLens-50K scaling protocol before official CF-loss LETTER assets are available |
| I07-RQKmeans | ready | `logs/scaling_microlens1m_large_drop/I07-rkmeans/tiger_compat_balanced/` | Balanced RQ-KMeans tokenizer matching the previous MicroLens-50K I07 protocol |

I04 SemID title-hierarchy stats:

| Field | Value |
|---|---:|
| items | 88,866 |
| path length | 4 |
| per-position unique nodes | `[64, 10527, 8, 1574]` |
| unique manifest tokens | 12,173 |
| unique prefixes before leaf | 43,144 |
| max leaf index | 1,573 |

I07 Balanced RQ-KMeans stats:

| Field | Value |
|---|---:|
| items | 88,866 |
| semantic code length | 3 |
| decoder code length | 4 |
| per-position sizes | `[256, 256, 256, 347]` |
| raw unique semantic IDs | 17,651 |
| raw collision rate | 0.801375 |
| max raw collision | 347 |
| first-level used codes | 256 |
| first-level usage rate | 1.0000 |
| dedup suffix | yes |
| dedup max suffix | 346 |

## Next Steps

| Order | Task | Status |
|---:|---|---|
| 1 | Build I07 RQ-Kmeans tokenizer from MicroLens-1M T5 embeddings | done |
| 2 | Build I05 LETTER-compatible tokenizer assets matching the previous MicroLens-50K scaling protocol | done |
| 3 | Rebuild I04 SemID tokenizer assets from title-derived hierarchy | done |
| 4 | Train/export small/base/large decoders for I04/I05/I07 | I05 small/base running; I04/I07 and I05 large waiting for free GPUs |
| 5 | Compare MicroLens-50K vs MicroLens-1M large-drop trend | pending |

## Decoder Runs

| ID | Model | GPU | Status | Last observed progress | Run directory |
|---|---|---:|---|---|---|
| I05-LETTER-compatible | M01-t5s | 0 | training | `35649/80000`, about `6.02 it/s` | `logs/scaling_microlens1m_large_drop/I05-letter/M01-t5s/20260425_080151__gpu0/` |
| I05-LETTER-compatible | M02-t5b | 1 | training | `10262/80000`, about `1.76 it/s` | `logs/scaling_microlens1m_large_drop/I05-letter/M02-t5b/20260425_080246__gpu1/` |

## Live Resource Snapshot

Last checked: `2026-04-25 09:40 UTC` on `target-server`.

| GPU | Status |
|---:|---|
| 0 | occupied by this study: `I05-letter/M01-t5s` |
| 1 | occupied by this study: `I05-letter/M02-t5b` |
| 2 | occupied by another user |
| 3 | occupied by another user |
| 4 | occupied by another user |
| 5 | occupied by another user |
| 6 | occupied by another user |

Current clean-launch rule:

- Do not launch new decoder cells until a GPU is confirmed free.
- Use `I07-rkmeans/tiger_compat_balanced`, not the rejected `I07-rkmeans/tiger_compat` sklearn tokenizer.
- Queue monitor is running on `target-server` as `scripts/scaling/monitor_microlens1m_large_drop_queue.sh`.
- Monitor log: `logs/scaling_microlens1m_large_drop/launchers/monitor_queue.nohup`.
- Current monitor policy: check GPUs `0 1 3 4 5 6` every `300s`; only launch when a GPU has no compute process and memory usage is below `2000 MB`.

Queued decoder cells:

| Order | ID | Model | Compat dir |
|---:|---|---|---|
| 1 | I04-SemID | M01-t5s | `tiger_compat_title_hierarchy` |
| 2 | I04-SemID | M02-t5b | `tiger_compat_title_hierarchy` |
| 3 | I04-SemID | M03-t5l | `tiger_compat_title_hierarchy` |
| 4 | I05-LETTER-compatible | M03-t5l | `tiger_compat` |
| 5 | I07-RQKmeans | M01-t5s | `tiger_compat_balanced` |
| 6 | I07-RQKmeans | M02-t5b | `tiger_compat_balanced` |
| 7 | I07-RQKmeans | M03-t5l | `tiger_compat_balanced` |

## Notes

- The full official LETTER CF-loss tokenizer would require `SASRec_item_embed.pkl` for `microlens_1m`. The current previous MicroLens-50K scaling logs used `logs/scaling_ml50k/I05-letter/tiger_compat`, whose `per_pos_sizes` are `[1, 1, 55, 256]`, i.e. the LETTER-compatible index generated under the existing scaling protocol. This study should first match that previous protocol for a clean 50K-to-1M comparison.
- Directly rebuilding I04 from `LLM_RecSys_ID` category metadata is invalid for MicroLens-1M: the metadata only contains a single root category and collapses to `sem_id_dim=1, codebook_size=88866`. This run therefore uses the same MicroLens-specific title-derived SemID construction principle documented for the previous MicroLens-50K protocol.
- A first I07 attempt using sklearn MiniBatchKMeans finished quickly but is rejected for the clean study: raw unique semantic IDs were only `21062/88866`, with `max_collision=1660`. The previous MicroLens-50K I07 used Balanced RQ-KMeans and had `max_collision=36`, so the 1M clean tokenizer is being rebuilt with the balanced backend in `tiger_compat_balanced/`.
