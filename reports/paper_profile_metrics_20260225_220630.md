# Paper-profile metrics snapshot (20260225_220630 UTC)

Dataset: SETRec splits from `third_party/SETRec/data/`.
Logs: `/data/xqp_data/RecSys26/logs/repro_paper/**`.

## Completed runs

### SETRec (T5-small)
- beauty: exit=0 run=`logs/repro_paper/setrec/beauty/20260225_175518/run.log`
  - All: R@5=0.0066 R@10=0.0109 N@5=0.0054 N@10=0.0070 (best beta=0.3)
  - train_end_epoch=8.56 (early-stop on eval_loss)
  - paper(ref): R@5=0.0104 R@10=0.0167 N@5=0.0085 N@10=0.0108
- toys: exit=0 run=`logs/repro_paper/setrec/toys/20260225_181726/run.log`
  - All: R@5=0.0115 R@10=0.0177 N@5=0.0088 N@10=0.0111 (best beta=0.7)
  - train_end_epoch=16.29 (early-stop on eval_loss)
  - paper(ref): R@5=0.0116 R@10=0.0188 N@5=0.0095 N@10=0.0120
- sports: exit=0 run=`logs/repro_paper/setrec/sports/20260225_192305/run.log`
  - All: R@5=0.0096 R@10=0.0154 N@5=0.0093 N@10=0.0113 (best beta=1.0)
  - train_end_epoch=8.72 (early-stop on eval_loss)
  - paper(ref): R@5=0.0114 R@10=0.0185 N@5=0.0101 N@10=0.0126
- steam: exit=0 run=`logs/repro_paper/setrec/steam/20260225_200448/run.log`
  - All: R@5=0.0196 R@10=0.0329 N@5=0.0232 N@10=0.0270 (best beta=0.5)
  - train_end_epoch=2.32 (early-stop on eval_loss)
  - paper(ref): R@5=0.0216 R@10=0.0383 N@5=0.0254 N@10=0.0308

### DiffGRM
- beauty: exit=0 run=`logs/repro_paper/diffgrm/beauty/20260225_163803/run.log`
  - Test: recall@5=0.01138 recall@10=0.01968 ndcg@5=0.00746 ndcg@10=0.01013
- sports: exit=0 run=`logs/repro_paper/diffgrm/sports/20260225_203158/run.log`
  - Test: recall@5=0.02085 recall@10=0.03248 ndcg@5=0.01324 ndcg@10=0.01700

### RPG
- beauty: exit=0 run=`logs/repro_paper/rpg/beauty/20260225_163818/run.log`
  - Test: recall@5=0.00246 recall@10=0.00523 ndcg@5=0.00177 ndcg@10=0.00262
- toys: exit=0 run=`logs/repro_paper/rpg/toys/20260225_212405/run.log`
  - Test: recall@5=0.00604 recall@10=0.00992 ndcg@5=0.00352 ndcg@10=0.00479

### SEATER
- beauty: exit=0 run=`logs/repro_paper/seater/beauty/20260225_164046/run.log`
  - Test: recall@20=0.0209 recall@50=0.0372 ndcg@20=0.0109 ndcg@50=0.0149
- toys: exit=0 run=`logs/repro_paper/seater/toys/20260225_212405/run.log`
  - Test: recall@20=0.0212 recall@50=0.0347 ndcg@20=0.0101 ndcg@50=0.0136

### EAGER
- beauty: exit=0 run=`logs/repro_paper/eager/beauty/20260225_163733/run.log`
  - last_eval(step=60000): R@5=0.0036 NDCG@5=0.0026 (topk=5)

### ETEGRec
- beauty: exit=0 run=`logs/repro_paper/etegrec/beauty/20260225_164100/run.log`
  - Test: recall@1=0.001557 recall@5=0.003425 ndcg@5=0.002545

## Failed / needs rerun
- DiffGRM toys: exit=1 run=`logs/repro_paper/diffgrm/toys/20260225_200532/run.log` (OOM during test beam search on GPU0)
- LLM_ID SID beauty: exit=1 run=`logs/repro_paper/llm_id/sid/beauty/20260225_191139/run.log` (CUDA illegal memory access; likely whole_word_id overflow)
