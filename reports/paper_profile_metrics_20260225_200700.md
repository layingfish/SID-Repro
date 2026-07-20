# Paper-profile metrics snapshot (20260225_200700 UTC)

Dataset: SETRec splits from `third_party/SETRec/data/`.
Logs: `/data/xqp_data/RecSys26/logs/repro_paper/**`.

## Completed runs

### SETRec (T5-small)
- beauty: R@5=0.0066 R@10=0.0109 N@5=0.0054 N@10=0.0070 (best beta=0.3)
  - log: `logs/repro_paper/setrec/beauty/20260225_175518/run.log`
  - paper(ref): R@5=0.0106 R@10=0.0161 N@5=0.0083 N@10=0.0103
- toys: R@5=0.0115 R@10=0.0177 N@5=0.0088 N@10=0.0111 (best beta=0.7)
  - log: `logs/repro_paper/setrec/toys/20260225_181726/run.log`
  - paper(ref): R@5=0.0110 R@10=0.0189 N@5=0.0089 N@10=0.0118
- sports: R@5=0.0096 R@10=0.0154 N@5=0.0093 N@10=0.0113 (best beta=1.0)
  - log: `logs/repro_paper/setrec/sports/20260225_192305/run.log`

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
- beauty: topk=5 Recall@5≈0.0036 NDCG@5≈0.0026
  - log: `logs/repro_paper/eager/beauty/20260225_163733/run.log`

### ETEGRec
- beauty: recall@1=0.001557 recall@5=0.003425 ndcg@5=0.002545
  - log: `logs/repro_paper/etegrec/beauty/20260225_164100/run.log`

## Running / queued
- SETRec steam: `logs/repro_paper/setrec/steam/latest/run.log`
- DiffGRM toys: `logs/repro_paper/diffgrm/toys/latest/run.log`

## Protocol notes
- SETRec / EAGER: evaluation only counts users with non-empty `testing_dict`.
- DiffGRM / RPG (SETRec dataset class): only keeps users with non-empty val/test when building sequences.
