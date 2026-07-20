# SETRec alignment check (20260225_223859 UTC)

Reference: SETRec paper Table (T5), extracted to `reports/setrec_paper_overall_t5_table.json`.

## beauty
- run: `/data/xqp_data/RecSys26/logs/repro_paper/setrec/beauty/latest`
- ours(all): R@5=0.0066 R@10=0.0109 N@5=0.0054 N@10=0.0070 (best_beta=0.3)
- paper(all): R@5=0.0106 R@10=0.0161 N@5=0.0083 N@10=0.0103
  - r@5: delta=-0.0040 (-37.7%)
  - r@10: delta=-0.0052 (-32.3%)
  - n@5: delta=-0.0029 (-34.9%)
  - n@10: delta=-0.0033 (-32.0%)

## toys
- run: `/data/xqp_data/RecSys26/logs/repro_paper/setrec/toys/latest`
- ours(all): R@5=0.0115 R@10=0.0177 N@5=0.0088 N@10=0.0111 (best_beta=0.7)
- paper(all): R@5=0.0110 R@10=0.0189 N@5=0.0089 N@10=0.0118
  - r@5: delta=+0.0005 (+4.5%)
  - r@10: delta=-0.0012 (-6.3%)
  - n@5: delta=-0.0001 (-1.1%)
  - n@10: delta=-0.0007 (-5.9%)

## sports
- run: `/data/xqp_data/RecSys26/logs/repro_paper/setrec/sports/latest`
- ours(all): R@5=0.0096 R@10=0.0154 N@5=0.0093 N@10=0.0113 (best_beta=1.0)
- paper(all): R@5=0.0114 R@10=0.0185 N@5=0.0101 N@10=0.0126
  - r@5: delta=-0.0018 (-15.8%)
  - r@10: delta=-0.0031 (-16.8%)
  - n@5: delta=-0.0008 (-7.9%)
  - n@10: delta=-0.0013 (-10.3%)

## steam
- run: `/data/xqp_data/RecSys26/logs/repro_paper/setrec/steam/latest`
- ours(all): R@5=0.0196 R@10=0.0329 N@5=0.0232 N@10=0.0270 (best_beta=0.5)
- paper(all): R@5=0.0216 R@10=0.0383 N@5=0.0254 N@10=0.0308
  - r@5: delta=-0.0020 (-9.3%)
  - r@10: delta=-0.0054 (-14.1%)
  - n@5: delta=-0.0022 (-8.7%)
  - n@10: delta=-0.0038 (-12.3%)

