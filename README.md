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

The scripts under `scripts/` are the canonical launchers. The research questions
are organized as follows:

- `RQ1`: overall benchmark under a unified generative recommendation framework.
- `RQ2`: codebook utilization and its relationship to recommendation quality.
- `RQ3`: scaling behavior with respect to SID length and decoder backbone size.
- `RQ4`: item-semantics preservation and geometry analysis.

The submitted hyperparameters are recorded under `configs/`. Use
`configs/experiment/*.yaml` as the RQ-level protocol index, and use
`configs/data/`, `configs/tokenizer/`, and `configs/decoder/` for the concrete
data, SID tokenizer, and decoder settings. The manifest-based decoder launcher
also uses the GIN files under `baselines/decoder/configs/`.

Pass the required data, tokenizer, checkpoint, and output paths through
command-line arguments.

```bash
bash scripts/run_data.sh --help
bash scripts/run_rq1.sh --help
bash scripts/run_rq2.sh --help
bash scripts/run_rq3.sh --help
bash scripts/run_rq4.sh --help
```

Prepare data:

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

Run configured launchers with environment variables:

```bash
RUN_RQ1_ARGS='evaluate --pred <predictions.jsonl> --dataset <dataset> --output <metrics.json>' \
RUN_RQ2_ARGS='--input <ids.npy> --output_dir <output-dir> --method_family <family> --paper_variant <variant> --implementation_variant <variant>' \
bash scripts/run_all.sh
```
