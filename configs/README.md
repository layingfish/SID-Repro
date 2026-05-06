# Experiment Configurations

This directory records the hyperparameters used for the submitted experiments.
Machine-specific paths are expressed with placeholders such as `${DATA_ROOT}`,
`${MODEL_ROOT}`, `${RUN_ROOT}`, and `${TOKENIZER_ROOT}`.

The launchers under `scripts/` remain the canonical entry points. These files are
intended to make the experimental protocol explicit and auditable.

## Files

- `data/setrec_splits.yaml`: data split and text-embedding protocol.
- `tokenizer/main_table_tokenizers.yaml`: method-native SID, encoder, decoder, and export settings for the main table.
- `tokenizer/length_scaling_tokenizers.yaml`: SID-length study settings.
- `decoder/main_table_decoder.yaml`: controlled manifest-based T5 decoder settings for RQ3 and additional alignment studies, not the RQ1 main table.
- `decoder/backbone_scaling.yaml`: small/base/large scaling settings.
- `experiment/rq1_overall.yaml`: RQ1 overall benchmark protocol.
- `experiment/rq2_codebook_usage.yaml`: RQ2 codebook-utilization protocol.
- `experiment/rq3_scaling.yaml`: RQ3 length/backbone scaling protocol.
- `experiment/rq4_geometry.yaml`: RQ4 neighborhood-preservation protocol.

RQ1 uses method-native training and decoding from the corresponding reference
implementations. The shared part of RQ1 is the data adapter, canonical item-id
space, JSONL prediction export, and final evaluator.

The decoder GIN files used by the controlled manifest-based decoder wrapper are
under `baselines/decoder/configs/`.
