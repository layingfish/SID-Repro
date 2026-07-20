# Paper-profile metrics snapshot (20260225_230813 UTC)

Dataset: SETRec splits from `third_party/SETRec/data/`.
Logs: `/data/xqp_data/RecSys26/logs/repro_paper/**`.

## Latest per method/domain

### diffgrm
- beauty: exit=0 run=`logs/repro_paper/diffgrm/beauty/latest/run.log`
  - Test: recall@5=0.011377613776137762 recall@10=0.01968019680196802 ndcg@5=0.007457675657442428 ndcg@10=0.010130044654799856
- sports: exit=0 run=`logs/repro_paper/diffgrm/sports/latest/run.log`
  - Test: recall@5=0.020854700854700856 recall@10=0.03247863247863248 ndcg@5=0.01324312064382765 ndcg@10=0.017004373226410303
- steam: exit=? run=`logs/repro_paper/diffgrm/steam/latest/run.log`
- toys: exit=? run=`logs/repro_paper/diffgrm/toys/latest/run.log`

### eager
- beauty: exit=0 run=`logs/repro_paper/eager/beauty/latest/run.log`
  - [EAGER][SETRec][beauty] step=60000 lr=0.000183 loss=0.6026 contra=0.0000 P@5=0.0007 R=0.0036 HR=0.0036 NDCG=0.0026
- toys: exit=0 run=`logs/repro_paper/eager/toys/latest/run.log`
  - [EAGER][SETRec][toys] step=60000 lr=0.000183 loss=0.3228 contra=0.0000 P@5=0.0006 R=0.0031 HR=0.0031 NDCG=0.0021

### etegrec
- beauty: exit=0 run=`logs/repro_paper/etegrec/beauty/latest/run.log`
  - Test: recall@1=0.001557 recall@5=0.003425 ndcg@5=0.002545
- toys: exit=0 run=`logs/repro_paper/etegrec/toys/latest/run.log`
  - Test: recall@1=0.00054 recall@5=0.001079 ndcg@5=0.000821

### llm_id/sid
- beauty: exit=? run=`logs/repro_paper/llm_id/sid/beauty/latest/run.log`
  - Test(last): hit@5=0.006150061500615006 hit@10=0.007995079950799507 ndcg@5=0.004784527717065753 ndcg@10=0.005413601427874295

### rpg
- beauty: exit=0 run=`logs/repro_paper/rpg/beauty/latest/run.log`
  - Test: recall@5=0.0024600245524197817 recall@10=0.0052275522612035275 ndcg@5=0.0017675908748060465 ndcg@10=0.002624680520966649
- toys: exit=0 run=`logs/repro_paper/rpg/toys/latest/run.log`
  - Test: recall@5=0.006039689294993877 recall@10=0.009922347031533718 ndcg@5=0.003521529957652092 ndcg@10=0.00479003693908453

### seater
- beauty: exit=0 run=`logs/repro_paper/seater/beauty/latest/run.log`
  - Test?: recall@20=0.0209 recall@50=0.0372 ndcg@20=0.0109 ndcg@50=0.0149
- sports: exit=? run=`logs/repro_paper/seater/sports/latest/run.log`
  - Test?: recall@20=0.0299 recall@50=0.0527 ndcg@20=0.0166 ndcg@50=0.0227
- toys: exit=0 run=`logs/repro_paper/seater/toys/latest/run.log`
  - Test?: recall@20=0.0212 recall@50=0.0347 ndcg@20=0.0101 ndcg@50=0.0136

### setrec
- beauty: exit=0 run=`logs/repro_paper/setrec/beauty/latest/run.log`
  - All: R@5=0.0066 R@10=0.0109 N@5=0.0054 N@10=0.007 (best_beta=0.3)
- sports: exit=0 run=`logs/repro_paper/setrec/sports/latest/run.log`
  - All: R@5=0.0096 R@10=0.0154 N@5=0.0093 N@10=0.0113 (best_beta=1.0)
- steam: exit=0 run=`logs/repro_paper/setrec/steam/latest/run.log`
  - All: R@5=0.0196 R@10=0.0329 N@5=0.0232 N@10=0.027 (best_beta=0.5)
- toys: exit=0 run=`logs/repro_paper/setrec/toys/latest/run.log`
  - All: R@5=0.0115 R@10=0.0177 N@5=0.0088 N@10=0.0111 (best_beta=0.7)

