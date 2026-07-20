# RecSys26 operational runbook

This runbook describes the organized source workspace. Historical commands in
`Process.md` may still show the former server-specific project root; they are
provenance records, not current path requirements.

## 1. Initialize paths

From the project root:

```bash
source scripts/env.sh
python3 scripts/verify_source_layout.py --syntax
```

`scripts/env.sh` resolves `RECSYS26_ROOT` from its own location. Override
`DATA_ROOT`, `MODEL_ROOT`, `ENV_ROOT`, `RUN_ROOT`, or `TOKENIZER_ROOT` when the
resources live on another volume.

## 2. Source versus external resources

Required source:

- `baselines/`, `configs/`, `experiments/`, `pipeline/`, `requirements/`
- `scripts/run_*.sh`
- `third_party_clean/` and the `third_party` compatibility link
- `patches/` and `repos_manifest.tsv`

External resources, downloaded or generated separately:

- `${DATA_ROOT}`: raw datasets, SETRec splits, adapters, and embeddings
- `${MODEL_ROOT}`: Hugging Face and PyTorch model caches
- `${ENV_ROOT}`: Conda environments
- `${RUN_ROOT}`: logs, checkpoints, predictions, and metrics
- `${TOKENIZER_ROOT}`: trained SID/tokenizer artifacts

## 3. Environment

Portable release environment:

```bash
conda env create -f requirements/environment.yml
conda activate recsys26-anon
pip install -r requirements/requirements.txt
```

Historical per-repository environments:

```bash
source scripts/conda_activate.sh recsys26_setrec
source scripts/conda_activate.sh recsys26_rqvae
source scripts/conda_activate.sh recsys26_letter
```

The environment specifications under `patches/*_env.yml` remain available for
method-specific reproduction.

## 4. Canonical experiment entry points

```bash
bash scripts/run_data.sh --help
bash scripts/run_rq1.sh --help
bash scripts/run_rq2.sh --help
bash scripts/run_rq3.sh --help
bash scripts/run_rq4.sh --help
```

- RQ1: complete method-native recommendation systems
- RQ2: codebook utilization
- RQ3: SID-length and decoder-backbone scaling
- RQ4: SID geometry and semantic-neighborhood preservation

Protocol values are frozen under `configs/`; the launchers pass paths through
CLI arguments or environment variables rather than assuming a machine-specific
root.

## 5. Evaluation

Use the canonical evaluator through the RQ1 launcher:

```bash
bash scripts/run_rq1.sh evaluate \
  --pred "${RUN_ROOT}/example/pred_topk.jsonl" \
  --dataset microlens_50k \
  --mode full \
  --data_dir "${DATA_ROOT}/setrec_data" \
  --top_n 5,10 \
  --warm_only \
  --output "${RUN_ROOT}/example/metrics_full.json"
```

Strict validation is enabled by default. Use `--non_strict` only for an
explicit diagnostic run.

## 6. Legacy workspace

The following are retained for provenance but are not the preferred API:

- `legacy/scripts/`: former scaling, from-scratch, smoke, and queue launchers
- `pred_exports/` historical prediction artifacts
- `archive/` pre-consolidation backups and reference snapshots
- `third_party_clean/` dirty worktrees containing documented adaptations

Do not delete or silently rewrite `Process.md`, `reports/`, or `patches/`; they
are part of the reproducibility audit trail.
