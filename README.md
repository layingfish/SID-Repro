# RecSys26 Source Workspace

This workspace contains the source code used for the RecSys 2026 experiments.
Datasets, pretrained models, environments, checkpoints, and run outputs are
external resources and are intentionally not part of the canonical source
layout.

## Canonical source layout

| Path | Purpose |
| --- | --- |
| `baselines/` | Source snapshots for the compared recommendation methods |
| `configs/` | Frozen data, tokenizer, decoder, and experiment protocols |
| `experiments/` | RQ1–RQ4 experiment entry points |
| `pipeline/` | Shared data, tokenizer, decoder, export, and evaluation code |
| `requirements/` | Reproducible Python/Conda environment specifications |
| `scripts/run_*.sh` | Canonical user-facing launchers |
| `docs/` | Resource provisioning and supplementary-experiment plans |
| `papers/RecSys_2026_paper_777.pdf` | Submitted-paper snapshot used to identify reusable results |
| `third_party_clean/` | Upstream Git worktrees used to audit and develop adaptations |
| `third_party` | Compatibility link to `third_party_clean/` |
| `patches/` | Explicit patches applied to upstream repositories |
| `reports/`, `*.md` | Experiment records and verified result summaries |
| `legacy/scripts/` | Historical smoke, queue, scaling, and one-off launchers |
| `archive/` | Historical backups, reference snapshots, and migrated outputs |

Historical launchers are retained under `legacy/scripts/` for experiment
provenance. New runs should start from `scripts/run_data.sh` and
`scripts/run_rq{1,2,3,4}.sh`.

## External resources

The exact download list and placement for the planned RQ2/RQ4 supplementary
experiments are documented in
[`docs/EXTERNAL_RESOURCES_AND_SUPPLEMENTARY_EXPERIMENTS.md`](docs/EXTERNAL_RESOURCES_AND_SUPPLEMENTARY_EXPERIMENTS.md).

The following locations may be absent in a source-only checkout:

- `datasets/` or a custom `DATA_ROOT`
- `models/` or a custom `MODEL_ROOT`
- `envs/` or a custom `ENV_ROOT`
- `logs/` or a custom `RUN_ROOT`
- tokenizer/checkpoint outputs under a custom `TOKENIZER_ROOT`

Set the workspace defaults with:

```bash
source scripts/env.sh
```

Every location can be overridden before sourcing:

```bash
export DATA_ROOT=/path/to/data
export MODEL_ROOT=/path/to/models
export ENV_ROOT=/path/to/conda/envs
export RUN_ROOT=/path/to/runs
export TOKENIZER_ROOT=/path/to/tokenizer_outputs
source scripts/env.sh
```

## Validate the source checkout

The layout check treats datasets and models as optional, but requires all
canonical source modules and upstream worktrees:

```bash
python3 scripts/verify_source_layout.py --syntax
```

Use `--strict-resources` only after external resources have been provisioned.

## Environment

Create the portable release environment with:

```bash
conda env create -f requirements/environment.yml
conda activate recsys26-anon
pip install -r requirements/requirements.txt
```

The historical per-method environments can still be activated with:

```bash
source scripts/conda_activate.sh <env_name>
```

## Canonical launchers

```bash
bash scripts/run_data.sh --help
bash scripts/run_rq1.sh --help
bash scripts/run_rq2.sh --help
bash scripts/run_rq3.sh --help
bash scripts/run_rq4.sh --help
```

Examples for data preparation and the RQ-specific protocols are documented in
`RUNBOOK.md` and `configs/README.md`.

## Provenance

- `repos_manifest.tsv` records the exact upstream commits.
- `Process.md` is the chronological experiment log.
- `reports/eval_protocol_lock.md` records the locked evaluation protocol.
- `archive/pre_organize_20260720/` contains entry points preserved before the
  source-layout consolidation.
