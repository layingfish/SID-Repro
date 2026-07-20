# MicroLens-1M VG-Style Scaling

## Goal

Run the MicroLens-1M scaling check under the same short VG main-table training recipe, instead of the previous MicroLens-50K `80k` scaling recipe.

Target IDs:

| ID | Method | Tokenizer source |
|---|---|---|
| I04 | SemID | `logs/scaling_microlens1m_large_drop/I04-semid/tiger_compat_title_hierarchy/` |
| I05 | LETTER-compatible | `logs/scaling_microlens1m_large_drop/I05-letter/tiger_compat/` |
| I07 | RQ-Kmeans | `logs/scaling_microlens1m_large_drop/I07-rkmeans/tiger_compat_balanced/` |

Model sizes:

| Model | Backbone |
|---|---|
| M01-t5s | T5-small |
| M02-t5b | T5-base |
| M03-t5l | T5-large |

## Protocol

This run intentionally follows the VG main-table style instead of the earlier MicroLens-50K scaling style.

| Setting | Value |
|---|---|
| dataset | `microlens_1m` |
| user tokens | `0` |
| user conditioning | disabled |
| iterations | `5000` |
| checkpoint | final `checkpoint_4999.pt` / `checkpoint_5000.pt` depending on script naming |
| learning rate | `3e-4` |
| weight decay | `0.035` |
| export beam | `64` |
| export return sequences | `64` |
| export top-k items | `20` |
| export scope | `warm_only_export`: only users whose test targets contain warm items |
| eval | `full` and `loo`, both `warm_only`, `top_n=5,10` |

Batching follows the existing VG alignment branch in `scripts/scaling/train_with_manifest.py`:

| Model | batch size | grad accumulation | effective examples/update |
|---|---:|---:|---:|
| T5-small | 64 | 4 | 256 |
| T5-base | 32 | 8 | 256 |
| T5-large | 16 | 16 | 256 |

Export batching was increased after confirming 48GB GPU headroom:

| Model | export batch size | warm-only users | export batches |
|---|---:|---:|---:|
| T5-small | 96 | 172724 | 1800 |
| T5-base | 48 | 172724 | 3599 |
| T5-large | 16 | 172724 | 10796 |

Important correction:

- The first warm-only export filter used `SeqData.itemId_fut`, which retained only `101512` users.
- `unified_eval.py --mode full --warm_only` actually evaluates `testing_dict.npy` after dropping cold GT items, which retains `172724` users.
- `scripts/scaling/export_with_manifest.py` was fixed to use the same `testing_dict.npy + warm_item.npy` source as `unified_eval.py`, and export resume is enabled in `scripts/scaling/run_microlens1m_vgstyle_export_eval.sh`.
- The earlier `101512`-user exports are invalid for full warm-only metrics, but their valid predictions are reused via `--resume` while filling the missing users.

## Stopped Previous Run

Stopped the previous `80k` MicroLens-1M scaling run:

| Process | Status |
|---|---|
| `I05-letter/M01-t5s/20260425_080151__gpu0` | stopped |
| `I05-letter/M02-t5b/20260425_080246__gpu1` | stopped |
| `monitor_microlens1m_large_drop_queue.sh` | stopped |

## Current Runs

Log root:

`/data/xqp_data/RecSys26/logs/scaling_microlens1m_vgstyle/`

| ID | Model | GPU | Status | Run directory |
|---|---|---:|---|---|
| I04-SemID | M01-t5s | 1 | training finished; fixed `172724`-user resume export running, `132875/172724` prediction rows observed, about `788/1800` export batches; no valid metrics yet | `logs/scaling_microlens1m_vgstyle/I04-semid/M01-t5s/20260425_103619__gpu0/` |
| I04-SemID | M02-t5b | 1 | training finished; old `101512`-user export invalid for full warm-only; queued to run fixed `172724`-user resume export immediately after I04-small on GPU1 | `logs/scaling_microlens1m_vgstyle/I04-semid/M02-t5b/20260425_103639__gpu1/` |
| I04-SemID | M03-t5l | 3 | training finished with `checkpoint_4999.pt`; original launcher did not enter export after script sync; first fixed export with batch `24` OOMed before writing predictions; export batch lowered to `16` and restarted on GPU3 | `logs/scaling_microlens1m_vgstyle/I04-semid/M03-t5l/20260425_121923__gpu3/` |
| I05-LETTER-compatible | M01-t5s | 0 | training finished; fixed `172724`-user warm-only export running, about `723/1800` export batches and `69405` prediction rows observed; no valid metrics yet | `logs/scaling_microlens1m_vgstyle/I05-letter/M01-t5s/20260425_155950__gpu0/` |

## Queue

Queue monitor:

`scripts/scaling/monitor_microlens1m_vgstyle_queue.sh`

Monitor log:

`logs/scaling_microlens1m_vgstyle/launchers/monitor_queue.nohup`

Queued cells:

| Order | ID | Model |
|---:|---|---|
| 1 | I04-SemID | M01-t5s |
| 2 | I04-SemID | M02-t5b |
| 3 | I04-SemID | M03-t5l |
| 4 | I05-LETTER-compatible | M01-t5s |
| 5 | I05-LETTER-compatible | M02-t5b |
| 6 | I05-LETTER-compatible | M03-t5l |
| 7 | I07-RQKmeans | M01-t5s |
| 8 | I07-RQKmeans | M02-t5b |
| 9 | I07-RQKmeans | M03-t5l |

## Notes

- This run is not directly comparable to the previous MicroLens-50K `80k` best-checkpoint scaling table.
- It is intended as a fast diagnostic to test whether VG-style short training produces reasonable small/base/large trends on MicroLens-1M.
- Export does not use `--vg_main_table_t5small_align` because that flag hardcodes `amazon23_vg_tiger_strict`; instead export explicitly uses `--domain microlens_1m`, `--force_num_user_tokens 0`, and VG-style beam settings.
- Original full export covered `515434` test users, which became `32215` batches at `batch_size=16`. This was stopped for the active I04 small/base runs.
- Warm-only export must keep `172724` users for `full warm_only`; `101512` was an invalid last-item-only approximation and should not be used for final metrics.
