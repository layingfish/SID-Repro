# Paper-profile metrics snapshot (20260226_134653 UTC)

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
- steam: exit=0 run=`logs/repro_paper/diffgrm/steam/20260225_212913/run.log`
  - Test: recall@5=0.0246225 recall@10=0.0413657 ndcg@5=0.0155365 ndcg@10=0.0209115

### eager
- beauty: exit=0 run=`logs/repro_paper/eager/beauty/20260225_163733/run.log`
  - [EAGER][SETRec][beauty] step=60000 lr=0.000183 loss=0.6026 contra=0.0000 P@5=0.0007 R=0.0036 HR=0.0036 NDCG=0.0026
- toys: exit=0 run=`logs/repro_paper/eager/toys/20260225_212355/run.log`
  - [EAGER][SETRec][toys] step=60000 lr=0.000183 loss=0.3228 contra=0.0000 P@5=0.0006 R=0.0031 HR=0.0031 NDCG=0.0021
- sports: exit=0 run=`logs/repro_paper/eager/sports/20260226_090019/run.log`
  - [EAGER][SETRec][sports] step=60000 lr=0.000183 loss=0.4209 contra=0.0000 P@5=0.0005 R@5=0.0026 HR@5=0.0026 NDCG@5=0.0018 P@10=0.0004 R@10=0.0044 HR@10=0.0044 NDCG@10=0.0024
- steam: exit=0 run=`logs/repro_paper/eager/steam/20260226_032314/run.log`
  - [EAGER][SETRec][steam] step=60000 lr=0.000183 loss=6.5701 contra=0.0000 P@5=0.0043 R@5=0.0213 HR@5=0.0213 NDCG@5=0.0130 P@10=0.0037 R@10=0.0366 HR@10=0.0366 NDCG@10=0.0179

### etegrec
- beauty: exit=0 run=`logs/repro_paper/etegrec/beauty/20260225_164100/run.log`
  - Test: {'recall@1': 0.001557, 'recall@5': 0.003425, 'ndcg@5': 0.002545}
- toys: exit=0 run=`logs/repro_paper/etegrec/toys/20260225_212405/run.log`
  - Test: {'recall@1': 0.00054, 'recall@5': 0.001079, 'ndcg@5': 0.000821}
- sports: exit=0 run=`logs/repro_paper/etegrec/sports/20260226_093527/run.log`
  - Test: {'recall@1': 0.002859, 'recall@5': 0.007883, 'ndcg@5': 0.005305, 'recall@10': 0.011175, 'ndcg@10': 0.006337}
- steam: exit=? run=`logs/repro_paper/etegrec/steam/20260226_085436/run.log`

### llm_id/sid
- beauty: exit=0 run=`logs/repro_paper/llm_id/sid/beauty/20260225_221002/run.log`
  - Test(last): hit@5=0.004305043050430504 hit@10=0.007380073800738007 ncdg@5=0.0030896180899666857 ncdg@10=0.00409783993881318
- toys: exit=0 run=`logs/repro_paper/llm_id/sid/toys/20260226_024545/run.log`
  - Test(last): hit@5=0.006902502157031924 hit@10=0.008628127696289905 ncdg@5=0.005132406487385153 ncdg@10=0.005654002434842976
- sports: exit=0 run=`logs/repro_paper/llm_id/sid/sports/20260226_061022/run.log`
  - Test(last): hit@5=0.006153846153846154 hit@10=0.008888888888888889 ncdg@5=0.004781048116046662 ncdg@10=0.0056359745837309535
- steam: -

### rpg
- beauty: exit=0 run=`logs/repro_paper/rpg/beauty/20260225_163818/run.log`
  - Test: recall@5=0.00246002 recall@10=0.00522755 ndcg@5=0.00176759 ndcg@10=0.00262468
- toys: exit=0 run=`logs/repro_paper/rpg/toys/20260225_212405/run.log`
  - Test: recall@5=0.00603969 recall@10=0.00992235 ndcg@5=0.00352153 ndcg@10=0.00479004
- sports: exit=0 run=`logs/repro_paper/rpg/sports/20260226_080534/run.log`
  - Test: recall@5=0.00376068 recall@10=0.00461538 ndcg@5=0.00233889 ndcg@10=0.00260152
- steam: exit=0 run=`logs/repro_paper/rpg/steam/20260226_055019/run.log`
  - Test: recall@5=0.0208236 recall@10=0.037004 ndcg@5=0.0128194 ndcg@10=0.0179973

### seater
- beauty: exit=0 run=`logs/repro_paper/seater/beauty/20260225_164046/run.log`
  - Test: recall@20=0.0209 recall@50=0.0372 ndcg@20=0.0109 ndcg@50=0.0149
- toys: exit=0 run=`logs/repro_paper/seater/toys/20260225_212405/run.log`
  - Test: recall@20=0.0212 recall@50=0.0347 ndcg@20=0.0101 ndcg@50=0.0136
- sports: exit=0 run=`logs/repro_paper/seater/sports/20260225_230359/run.log`
  - Test: recall@20=0.0259 recall@50=0.0465 ndcg@20=0.0145 ndcg@50=0.02
- steam: exit=0 run=`logs/repro_paper/seater/steam/20260226_080744/run.log`
  - Test: recall@20=0.0686 recall@50=0.1309 ndcg@20=0.0419 ndcg@50=0.0608

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
- beauty: exit=0 run=`logs/repro_paper/tiger/beauty/20260225_233523/run.log`
  - Test: recall@5=0.00184502 recall@10=0.00461255 ndcg@5=0.000885343 ndcg@10=0.00175204 (D=4)
- toys: exit=0 run=`logs/repro_paper/tiger/toys/20260226_074209/run.log`
  - Test: recall@5=0.00301984 recall@10=0.00474547 ndcg@5=0.00140897 ndcg@10=0.00195737 (D=4)
- sports: exit=0 run=`logs/repro_paper/tiger/sports/20260226_080100/run.log`
  - Test: recall@5=0.00222222 recall@10=0.00598291 ndcg@5=0.00108023 ndcg@10=0.00231053 (D=4)
- steam: exit=0 run=`logs/repro_paper/tiger/steam/20260226_091635/run.log`
  - Test: recall@5=0.0280461 recall@10=0.0448363 ndcg@5=0.0174588 ndcg@10=0.0228219 (D=4)

### letter
- beauty: exit=0 run=`logs/repro_paper/letter/beauty/20260226_102944/run.log`
  - Test: recall@5=0.000922509 recall@10=0.00184502 ndcg@5=0.00052046 ndcg@10=0.000790804
- toys: exit=0 run=`logs/repro_paper/letter/toys/20260226_111326/run.log`
  - Test: recall@5=0.00129422 recall@10=0.00172563 ndcg@5=0.000711265 ndcg@10=0.000864935
- sports: exit=0 run=`logs/repro_paper/letter/sports/20260226_104855/run.log`
  - Test: recall@5=0.00444444 recall@10=0.00683761 ndcg@5=0.00329465 ndcg@10=0.00408405
- steam: exit=143 run=`logs/repro_paper/letter/steam/20260226_113958/run.log`

