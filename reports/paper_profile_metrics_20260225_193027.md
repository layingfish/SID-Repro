# Paper-profile metrics snapshot (20260225_193027 UTC)

Dataset: SETRec splits from `third_party/SETRec/data/`.
Logs: `/data/xqp_data/RecSys26/logs/repro_paper/**`.

## Completed runs

### SETRec (T5-small)
- beauty: R@5=0.0066 R@10=0.0109 N@5=0.0054 N@10=0.0070 (best beta=0.3)
  - log: `logs/repro_paper/setrec/beauty/20260225_175518/run.log`
  - paper(ref): R@5=0.0106 R@10=0.0161 N@5=0.0083 N@10=0.0103 (from `reports/metric_check_20260225_130735.md`)
- toys: R@5=0.0115 R@10=0.0177 N@5=0.0088 N@10=0.0111 (best beta=0.7)
  - log: `logs/repro_paper/setrec/toys/20260225_181726/run.log`
  - paper(ref): R@5=0.0110 R@10=0.0189 N@5=0.0089 N@10=0.0118 (from `reports/metric_check_20260225_130735.md`)

### DiffGRM
- beauty: recall@5=0.01138 recall@10=0.01968 ndcg@5=0.00746 ndcg@10=0.01013
  - log: `logs/repro_paper/diffgrm/beauty/20260225_163803/run.log`

### RPG
- beauty: recall@5=0.00246 recall@10=0.00523 ndcg@5=0.00177 ndcg@10=0.00262
  - log: `logs/repro_paper/rpg/beauty/20260225_163818/run.log`

### SEATER
- beauty: recall@20=0.0209 recall@50=0.0372 ndcg@20=0.0109 ndcg@50=0.0149
  - log: `logs/repro_paper/seater/beauty/20260225_164046/run.log`

### EAGER
- beauty: Recall@5≈0.0036 NDCG@5≈0.0026 (topk=5)
  - log: `logs/repro_paper/eager/beauty/20260225_163733/run.log`

### ETEGRec
- beauty: recall@1=0.001557 recall@5=0.003425 ndcg@5=0.002545
  - log: `logs/repro_paper/etegrec/beauty/20260225_164100/run.log`

## Running / queued
- SETRec sports: `logs/repro_paper/setrec/sports/latest/run.log` (re-run after补齐 warm/cold item files)
- LLM_RecSys_ID SID beauty: `logs/repro_paper/llm_id/sid/beauty/latest/run.log` (fixed launch; now running)

## Credibility / protocol notes
- SETRec / EAGER: evaluation 只统计 `testing_dict` 非空用户，避免把无label用户引入导致指标虚高。
- DiffGRM: evaluator 已修复 duplicate scoring / illegal sequence 过滤；日志会打印 `Using evaluator = DIFF_GRMEvaluator (fixed ...)`。
- SETRec sports: 原始仓库 `data/sports/` 缺少 `warm_item.npy`/`cold_item.npy`，已按“train 中出现过的 item 为 warm，其余为 cold”补齐。
