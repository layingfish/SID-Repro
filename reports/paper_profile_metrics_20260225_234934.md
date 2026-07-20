# Paper-profile metrics snapshot (20260225_234934 UTC)

Dataset: SETRec splits from `third_party/SETRec/data/`.
Logs: `/data/xqp_data/RecSys26/logs/repro_paper/**`.

## Latest per method/domain

### diffgrm
- beauty: exit=0 run=`logs/repro_paper/diffgrm/beauty/20260225_163803/run.log`
  - Test: recall@5=0.0113776 recall@10=0.0196802 ndcg@5=0.00745768 ndcg@10=0.01013
- toys: exit=0 run=`logs/repro_paper/diffgrm/toys/20260225_230135/run.log`
  - Test: recall@5=0.0150992 recall@10=0.0194133 ndcg@5=0.0111542 ndcg@10=0.0125517
- sports: exit=0 run=`logs/repro_paper/diffgrm/sports/20260225_203158/run.log`
  - Test: recall@5=0.0208547 recall@10=0.0324786 ndcg@5=0.0132431 ndcg@10=0.0170044
- steam: exit=? run=`logs/repro_paper/diffgrm/steam/20260225_212913/run.log`

### eager
- beauty: exit=0 run=`logs/repro_paper/eager/beauty/20260225_163733/run.log`
  - [EAGER][SETRec][beauty] step=60000 lr=0.000183 loss=0.6026 contra=0.0000 P@5=0.0007 R=0.0036 HR=0.0036 NDCG=0.0026
- toys: exit=0 run=`logs/repro_paper/eager/toys/20260225_212355/run.log`
  - [EAGER][SETRec][toys] step=60000 lr=0.000183 loss=0.3228 contra=0.0000 P@5=0.0006 R=0.0031 HR=0.0031 NDCG=0.0021
- sports: -
- steam: -

### etegrec
- beauty: exit=0 run=`logs/repro_paper/etegrec/beauty/20260225_164100/run.log`
  - Test: {'recall@1': 0.001557, 'recall@5': 0.003425, 'ndcg@5': 0.002545}
- toys: exit=0 run=`logs/repro_paper/etegrec/toys/20260225_212405/run.log`
  - Test: {'recall@1': 0.00054, 'recall@5': 0.001079, 'ndcg@5': 0.000821}
- sports: -
- steam: -

### llm_id/sid
- beauty: exit=? run=`logs/repro_paper/llm_id/sid/beauty/20260225_221002/run.log`
  - Test(last): hit@5=0.0036900369003690036 hit@10=0.006150061500615006 ncdg@5=0.0028054159358209415 ncdg@10=0.003608635734646935
- toys: -
- sports: -
- steam: -

### rpg
- beauty: exit=0 run=`logs/repro_paper/rpg/beauty/20260225_163818/run.log`
  - Test: recall@5=0.00246002 recall@10=0.00522755 ndcg@5=0.00176759 ndcg@10=0.00262468
- toys: exit=0 run=`logs/repro_paper/rpg/toys/20260225_212405/run.log`
  - Test: recall@5=0.00603969 recall@10=0.00992235 ndcg@5=0.00352153 ndcg@10=0.00479004
- sports: -
- steam: -

### seater
- beauty: exit=0 run=`logs/repro_paper/seater/beauty/20260225_164046/run.log`
  - Test: recall@20=0.0209 recall@50=0.0372 ndcg@20=0.0109 ndcg@50=0.0149
- toys: exit=0 run=`logs/repro_paper/seater/toys/20260225_212405/run.log`
  - Test: recall@20=0.0212 recall@50=0.0347 ndcg@20=0.0101 ndcg@50=0.0136
- sports: exit=0 run=`logs/repro_paper/seater/sports/20260225_230359/run.log`
  - Test: recall@20=0.0259 recall@50=0.0465 ndcg@20=0.0145 ndcg@50=0.02
- steam: -

### setrec
- beauty: exit=0 run=`logs/repro_paper/setrec/beauty/20260225_175518/run.log`
  - All: R@5=0.0066 R@10=0.0109 N@5=0.0054 N@10=0.007 (best_beta=0.3)
- toys: exit=0 run=`logs/repro_paper/setrec/toys/20260225_181726/run.log`
  - All: R@5=0.0115 R@10=0.0177 N@5=0.0088 N@10=0.0111 (best_beta=0.7)
- sports: exit=0 run=`logs/repro_paper/setrec/sports/20260225_192305/run.log`
  - All: R@5=0.0096 R@10=0.0154 N@5=0.0093 N@10=0.0113 (best_beta=1.0)
- steam: exit=0 run=`logs/repro_paper/setrec/steam/20260225_200448/run.log`
  - All: R@5=0.0196 R@10=0.0329 N@5=0.0232 N@10=0.027 (best_beta=0.5)

### tiger
- beauty: exit=? run=`logs/repro_paper/tiger/beauty/20260225_233523/run.log`
- toys: -
- sports: -
- steam: -

### letter
- beauty: -
- toys: -
- sports: -
- steam: -

