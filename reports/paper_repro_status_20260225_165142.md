# Paper-profile reproducibility status (${ts} UTC)

## Running paper-profile jobs (Beauty)
- EAGER: `logs/repro_paper/eager/beauty/20260225_163733/run.log`
- DiffGRM: `logs/repro_paper/diffgrm/beauty/20260225_163803/run.log`
- RPG: `logs/repro_paper/rpg/beauty/20260225_163818/run.log`
- SEATER: `logs/repro_paper/seater/beauty/20260225_164046/run.log`
- ETEGRec: `logs/repro_paper/etegrec/beauty/20260225_164100/run.log`

## Known issues from previous (non-paper) runs
- SETRec (T5) runs under `logs/repro/setrec/**` were using reduced settings (e.g. `n_sem=1`, `cutoff_len=64`, small AE), so metrics were ~40–50% below the SETRec paper table.
- EAGER had evaluation leakage in the earlier adapter (evaluating users without `testing_dict` labels). This was fixed in `third_party/EAGER/EAGER/train_rec_setrec.py` by restricting to `testing_dict` non-empty users and using `train+val` as history.
- LETTER was using a placeholder `.index.json` and (upstream) initialized from config only; `third_party/LETTER/LETTER-TIGER/finetune.py` was patched to load pretrained weights via `LETTER.from_pretrained(...)`.

## GPU snapshot
- `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv` at write-time:
  - 0: 45085/49140 MiB (vLLM)
  - 1: 9473/49140 MiB (DiffGRM + other)
  - 2: 4/49140 MiB (RPG tokenizer stage)
  - 3: 31396/49140 MiB (LLM_ID non-paper run still running)
  - 4: 8985/49140 MiB (SEATER + other)
  - 5: 3202/49140 MiB (EAGER)
  - 6: 12509/49140 MiB (ETEGRec + other)

## Next missing paper-profile runs (not started yet)
- SETRec (paper hyperparams; needs 4 GPUs)
- LLM_RecSys_ID variants (SID/CID/SemID/HID; needs 2 GPUs)
- LETTER (needs 2 GPUs + proper tokenizer-generated indices)
- TIGER (full paper alignment still needs RQ-VAE pretrain; decoder-only runner exists)

