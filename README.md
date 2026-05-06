# Anonymous RecSys 2026 Submission

This repository contains the code for reproducing the experiments in our RecSys 2026 submission.

## Environment

We provide the environment specification in:

- `requirements/environment.yml`
- `requirements/requirements.txt`

The environment can be created with:

```bash
conda env create -f requirements/environment.yml
conda activate recsys26-anon
pip install -r requirements/requirements.txt
```

## Reference

This project refers to the following public repositories:

- https://github.com/EdoardoBotta/RQ-VAE-Recommender
- https://github.com/ethan00si/seater_generative_retrieval
- https://github.com/Wenyueh/LLM-RecSys-ID
- https://github.com/HonghuiBao2000/LETTER
- https://github.com/facebookresearch/RPG_KDD2025
- https://github.com/snap-research/GRID
- https://github.com/westlake-repl/MicroLens
- https://github.com/Linxyhaha/SETRec
- https://github.com/RUCAIBox/ETEGRec
- https://github.com/liuzhao09/DiffGRM

## Running Experiments

The scripts under `scripts/` are the canonical launchers. The submitted
hyperparameters are recorded under `configs/`; the README summarizes the exact
protocol and the YAML files are the audit trail for all values used in the
submission.

```bash
bash scripts/run_data.sh --help
bash scripts/run_rq1.sh --help
bash scripts/run_rq2.sh --help
bash scripts/run_rq3.sh --help
bash scripts/run_rq4.sh --help
```

Set paths before running experiments:

```bash
export DATA_ROOT=/path/to/data
export MODEL_ROOT=/path/to/hf_models
export TOKENIZER_ROOT=/path/to/tokenizer_outputs
export RUN_ROOT=/path/to/runs
```

### Data preparation

The unified split and text-embedding protocol is in
`configs/data/setrec_splits.yaml`. The common setting is a 5-core filtered
SETRec-style time split with `split_ratio=0.17`, `val_multiplier=1.8`,
`min_train_len=2`, `item_shuffle_seed=2023`, and
`sentence-transformers/sentence-t5-base` item-text embeddings of dimension 768.

```bash
bash scripts/run_data.sh download --dataset microlens_50k --output_root data/raw
bash scripts/run_data.sh download --dataset amazon23_vg --output_root data/raw
YELP_OPEN_DATASET_URL='<official-yelp-json-archive-url>' bash scripts/run_data.sh download --dataset yelp --output_root data/raw --extract
bash scripts/run_data.sh download --dataset yelp --local_archive <path-to-yelp-dataset-archive> --output_root data/raw --extract
bash scripts/run_data.sh split --format microlens --input data/raw/microlens_50k/MicroLens-50k_pairs.csv --out_root data/setrec_data --dataset microlens_50k
bash scripts/run_data.sh split --format amazon23 --input data/raw/amazon23_vg/Video_Games.jsonl.gz --out_root data/setrec_data --dataset amazon23_vg
bash scripts/run_data.sh split --format yelp --input data/raw/yelp/extracted/yelp_academic_dataset_review.json --out_root data/setrec_data --dataset yelp
bash scripts/run_data.sh build-text --dataset microlens_50k --setrec_data_root data/setrec_data --microlens_titles data/raw/microlens_50k/MicroLens-50k_titles.csv
bash scripts/run_data.sh build-text --dataset amazon23_vg --setrec_data_root data/setrec_data --amazon_meta data/raw/amazon23_vg/meta_Video_Games.jsonl.gz
bash scripts/run_data.sh build-text --dataset yelp --setrec_data_root data/setrec_data --yelp_business data/raw/yelp/extracted/yelp_academic_dataset_business.json
bash scripts/run_data.sh build-embeddings --dataset microlens_50k --setrec_data_root data/setrec_data
```

The same `split`, `build-text`, and `build-embeddings` commands apply to
`microlens_100k`, `amazon23_vg`, and `yelp` after replacing the dataset-specific
raw input paths.

### RQ1: Overall benchmark

RQ1 compares representative SID designs under the unified decoder/evaluation
framework. The RQ-level protocol is in `configs/experiment/rq1_overall.yaml`;
the main tokenizer settings are in `configs/tokenizer/main_table_tokenizers.yaml`;
the decoder recipe is in `configs/decoder/main_table_decoder.yaml`.

Methods: `TIGER`, `T5-SemID`, `RPG`, `LETTER-TIGER`, `LETTER-LC-Rec`,
`SETRec`, `ETEGRec`, `SEATER`, `EAGER`, `DiffGRM`, `SASRec`, and `OneRec`.

Main hyperparameters:

| Component | Submitted setting |
| --- | --- |
| Reference text embedding | `sentence-transformers/sentence-t5-base`, 768-d |
| Common SID codebook size | 256 unless the reference method defines a different internal vocabulary |
| Main semantic code length | 3 semantic positions for the unified RQ-VAE/RQ-Kmeans-style setting; method-specific lengths are listed in `configs/tokenizer/main_table_tokenizers.yaml` |
| Unified decoder | `t5-small`, no UID tokens, `iterations=5000`, `learning_rate=3e-4`, `weight_decay=0.035` |
| Decoder batch schedule | `batch_size=64`, `gradient_accumulate_every=4`, bf16 AMP |
| Export | `beam_size=64`, `num_return_sequences=64`, `topk_items=20`, `batch_size=16` |
| Evaluation | full ranking mode, strict validation, `Recall@5/10` and `NDCG@5/10` |

Example manifest-based run:

```bash
bash scripts/run_rq1.sh train-decoder \
  --cached_ids_path ${TOKENIZER_ROOT}/<method>/cached_ids.npy \
  --tiger_config_path ${TOKENIZER_ROOT}/<method>/tiger_config.json \
  --gin_config baselines/decoder/configs/decoder_setrec_yelp_t5small_v2_lr3e4_paper.gin \
  --vg_main_table_t5small_align

bash scripts/run_rq1.sh export-decoder \
  --cached_ids_path ${TOKENIZER_ROOT}/<method>/cached_ids.npy \
  --tiger_config_path ${TOKENIZER_ROOT}/<method>/tiger_config.json \
  --decoder_ckpt ${RUN_ROOT}/<method>/checkpoint_4999.pt \
  --dataset_folder ${DATA_ROOT}/decoder \
  --domain amazon23_vg_tiger_strict \
  --hf_model_path ${MODEL_ROOT}/t5-small \
  --export_path ${RUN_ROOT}/<method>/pred_topk.jsonl \
  --beam_size 64 \
  --num_return_sequences 64 \
  --topk_items 20 \
  --batch_size 16 \
  --force_num_user_tokens 0

bash scripts/run_rq1.sh evaluate \
  --pred ${RUN_ROOT}/<method>/pred_topk.jsonl \
  --dataset amazon23_vg \
  --mode full \
  --data_dir ${DATA_ROOT}/setrec_data \
  --top_n 5,10 \
  --output ${RUN_ROOT}/<method>/metrics_full.json
```

### RQ2: Codebook utilization

RQ2 analyzes whether codebook utilization correlates with recommendation
performance. The protocol is in `configs/experiment/rq2_codebook_usage.yaml`.

Reported tokenizer metrics are computed per layer: `used_tokens`,
`dead_tokens`, `usage_rate`, `entropy`, `normalized_entropy`,
`effective_tokens`, `effective_usage_rate`, `gini`, `top1_mass`, and
`top5_mass`. Collision metrics include `unique_full_codes`, `collision_rate`,
`max_collision_group`, and `singleton_ratio`.

```bash
bash scripts/run_rq2.sh \
  --input ${TOKENIZER_ROOT}/<method>/cached_ids.npy \
  --output_dir ${RUN_ROOT}/rq2/<method> \
  --method_family <family> \
  --paper_variant <paper-variant> \
  --implementation_variant <implementation-variant> \
  --embedding_source sentence-t5-base \
  --embedding_dim 768 \
  --codebook_sizes 256,256,256 \
  --dedup_suffix
```

For balanced assignments such as RQ-Kmeans-style tokenizers, also pass
`--balanced_assignment`.

### RQ3: SID length and backbone scaling

RQ3 has two parts: SID length scaling and decoder-backbone scaling. The protocol
is in `configs/experiment/rq3_scaling.yaml`; tokenizer length settings are in
`configs/tokenizer/length_scaling_tokenizers.yaml`; backbone settings are in
`configs/decoder/backbone_scaling.yaml`.

Hyperparameter ranges:

| Study | Range |
| --- | --- |
| SID families | `RQ-VAE`, `RQ-Kmeans`, `OPQ` |
| SID lengths | `2, 3, 4, 6, 8, 12, 16` |
| Codebook width | 256 tokens per semantic position |
| RQ-Kmeans | balanced residual K-means, `max_iter=50`, `search_topk=32` |
| OPQ/PQ | `OPQ{length},IVF1,PQ{length}x8`, 8-bit sub-codebooks, PCA dim 256, inner-product Faiss metric |
| Decoder sizes | `t5-small`, `t5-base`, `t5-large` |
| Decoder schedule | `iterations=5000`, `learning_rate=3e-4`, `weight_decay=0.035`, no UID tokens |
| Decoder batch schedule | small `64 x grad_acc 4`, base `32 x grad_acc 8`, large `16 x grad_acc 16` |

Train/export/evaluate each SID length or backbone checkpoint with the same
`train-decoder`, `export-decoder`, and `evaluate` subcommands used in RQ1,
changing only the cached ID path, T5 backbone path, and output directory.

Inference-only diagnostics:

```bash
bash scripts/run_rq3.sh length-diagnostics \
  --study_root ${RUN_ROOT}/rq3_length \
  --data_dir ${DATA_ROOT}/setrec_data \
  --dataset amazon23_vg \
  --mode full \
  --methods rqvae,rqkmeans,opq \
  --lengths 2,3,4,6,8,12,16 \
  --top_n 5,10 \
  --output_dir ${RUN_ROOT}/rq3_length_diagnostics

bash scripts/run_rq3.sh capacity-diagnostics \
  --study_root ${RUN_ROOT}/rq3_length \
  --output_dir ${RUN_ROOT}/rq3_capacity_diagnostics \
  --methods rqvae,rqkmeans,opq \
  --lengths 2,3,4,6,8,12,16 \
  --data_dir ${DATA_ROOT}/setrec_data \
  --eval_dataset amazon23_vg \
  --dataset_folder ${DATA_ROOT}/decoder \
  --export_domain amazon23_vg_tiger_strict \
  --hf_model_path ${MODEL_ROOT}/t5-small \
  --top_ks 5,10 \
  --batch_size 32
```

### RQ4: SID geometry and item-semantics preservation

RQ4 measures whether SID neighborhoods preserve item-semantics neighborhoods in
the reference item-text embedding space. The protocol is in
`configs/experiment/rq4_geometry.yaml`.

Main settings:

| Component | Submitted setting |
| --- | --- |
| Reference space | `sentence-transformers/sentence-t5-base` item embeddings, cosine distance |
| Methods in the case study | `TIGER`, `SEATER`, `HowToIndex`, `LETTER`, `OPQ`, `RQ-Kmeans` |
| Main metrics | `Jaccard@20` over top-20 neighbors and `RBO@20` with persistence `p=0.9` |
| Pair/triplet samples | 100000 random pairs and 50000 random triplets |
| Anchor sampling | 128 anchors, top-200 anchor ranking pool, seed `20260423` |
| Visualization | 1 selected anchor, local context `k=120`, font scale `2.0` |
| SID distance | prefix distance for hierarchical/tree IDs; Hamming distance for OPQ/PQ IDs |

```bash
bash scripts/run_rq4.sh \
  --reference_embeddings ${DATA_ROOT}/setrec_data/amazon23_vg/amazon23_vg.emb-t5-tdcb.npy \
  --sid TIGER=${TOKENIZER_ROOT}/I02-TIGER.cached_ids.npy \
  --sid SEATER=${TOKENIZER_ROOT}/I03-SEATER.cached_ids.npy \
  --sid HowToIndex=${TOKENIZER_ROOT}/I04-SemID.cached_ids.npy \
  --sid LETTER=${TOKENIZER_ROOT}/I05-LETTER.cached_ids.npy \
  --sid OPQ=${TOKENIZER_ROOT}/I06-DiffGRM.cached_ids.npy \
  --sid RQ-Kmeans=${TOKENIZER_ROOT}/I07-RQKmeans.cached_ids.npy \
  --output_dir ${RUN_ROOT}/rq4_geometry \
  --pair_samples 100000 \
  --triplet_samples 50000 \
  --anchor_samples 128 \
  --neighbors_k 20 \
  --plot_anchors 1 \
  --plot_anchor_item_ids 18089 \
  --plot_context_k 120 \
  --plot_font_scale 2.0 \
  --paper_sid_metric native \
  --seed 20260423
```

### Running multiple RQs

`scripts/run_all.sh` forwards environment-variable argument strings to the RQ
launchers. This is useful for batch jobs after data, tokenizers, and checkpoints
have been prepared.

```bash
RUN_RQ1_ARGS='evaluate --pred <predictions.jsonl> --dataset amazon23_vg --mode full --data_dir data/setrec_data --top_n 5,10 --output <metrics.json>' \
RUN_RQ2_ARGS='--input <ids.npy> --output_dir <output-dir> --method_family <family> --paper_variant <variant> --implementation_variant <variant>' \
RUN_RQ3_ARGS='length-diagnostics --study_root <study-root> --output_dir <output-dir>' \
RUN_RQ4_ARGS='--reference_embeddings <emb.npy> --sid TIGER=<ids.npy> --output_dir <output-dir>' \
bash scripts/run_all.sh
```
