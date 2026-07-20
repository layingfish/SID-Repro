# RecSys26 Reproducibility — Process Log

本文件用于**实时记录**复现过程中的关键决策、命令、产物位置与结果（以远端运行日志为准，本地仅做索引与总结）。

## 0. 约束（必须遵守）

- **语言**：除命令/路径/报错外，说明使用简体中文。
- **GPU 约束**：
  - 训练/推理 **仅使用 GPU0/1/3/4**（避开 GPU2、GPU6；其它 GPU 默认不占用）。
  - **同时最多占用 4 张 GPU**（总并发 GPU 数 ≤ 4）。
- **目标指标（最重要）**：`R@5`、`R@10`、`N@5`、`N@10`（分别对应 Recall@K 与 NDCG@K）。其它指标（如 `R@20`/`N@20` 等）若官方脚本/统一评测顺手产出，也一并记录，但不作为主目标。
- **复现原则**：尽可能使用论文/官方仓库设置与代码；若对新数据集必须做默认选择（例如未知超参），在此文件中明确记录为“偏离点”并给出理由。

## 1. 环境与路径（远端为真实执行环境）

- 本地工作区：`/Users/chenyufei/Documents/RecSys26`
- 远端用户与主机：`qipeng@target-server`
- 远端 RecSys26 Root：`/data/xqp_data/RecSys26`
- 远端第三方代码（clean）：`/data/xqp_data/RecSys26/third_party_clean`
- 远端第三方代码（运行脚本主要指向）：`/data/xqp_data/RecSys26/third_party`

## 2. 数据集（已完成 splits）

我们当前需要在以下新数据集上复现全部方法：

- `amazon23_vg`
- `microlens_50k`
- `yelp`

已确认这些数据集的 SETRec-split（`training_dict.npy`/`validation_dict.npy`/`testing_dict.npy` 等）已经生成并落盘于：

- `qipeng@target-server:/data/xqp_data/RecSys26/third_party/SETRec/data/{amazon23_vg,microlens_50k,yelp}/`

> 注意：这些目录目前只保证 splits/映射文件齐全；尚需补齐后续方法依赖的
> `combine_tdcb_maps.npy`、`<dataset>.emb-t5-tdcb.npy`、`SASRec_item_embed.pkl` 等资产。

## 3. 任务追踪

- 任务清单：`tasks.csv`
- 规则：
  - 任何任务状态变化（TODO→RUNNING→DONE/FAILED/BLOCKED）都需要在本文件追加记录。
  - 若遇到我无法自主判断/需要你决策的问题：在 `tasks.csv` 将下一步更新为 **「暂停执行，向用户报告」**，并在本文件写清楚阻塞原因与待你确认的选项。

## 4. 进度流水（持续追加）

### 2026-04-24

- `RQ4` 最终口径确认：主问题是评估“不同 semantic ID 设计最终是否学到了 item semantics”，因此采用 `representation-level evaluation`，而不是 `controlled tokenizer-only ablation`。
- `RQ4` 主协议：
  - 保留六种方法各自原始/官方的 SID 生成方式；
  - 统一采用外部 reference space：`amazon23_vg.emb-t5-tdcb.npy`；
  - 统一以 `scripts/analyze_sid_geometry.py` 计算最终 `SID geometry` 与 reference geometry 的一致性；
  - 正文主表指标仍为 `Jaccard@20 + RBO@20`；
  - 当前正文主表已进一步切换为 `native SID distance`：
    - `I02/I03/I04/I05/I07 -> prefix`
    - `I06 -> hamming`
  - 各方法在几何分析前默认去掉 `dedup` 后缀位（当前 적용到 `I02/I03/I05/I06/I07`）。
- 不再采用“强行把全部方法迁移到统一 T5 source space”作为主 `RQ4` 方案，原因：
  - `I03-SEATER` 若改到 `T5` source，会与 `I02-TIGER` 的 content RQ-VAE 定义高度重合；
  - `I05-LETTER` 官方实现显式依赖 `cf_emb`，移除后不再是原始 LETTER；
  - `I04-SemID` 本质是 metadata/category hierarchy 的离散编码，不是 embedding quantizer。
- 因此当前六方法 `I02/I03/I04/I05/I06/I07` 仍可全部纳入 `RQ4` 正文主表，但论文中必须明确：
  - 我们比较的是各方法 `final SID geometry`；
  - 不是统一 source 下的 tokenizer-only controlled benchmark。
- 之前讨论的 `controlled T5-space tokenizer rebuild` 计划保留，但降级为 supplementary ablation，仅用于分析统一 source semantic space 下的量化器差异，不再作为正文 `RQ4` 主协议。
- 已更新 `scripts/analyze_sid_geometry.py`：
  - 新增 `native` 作为 `paper_sid_metric`
  - 新增方法级 `native metric override`
  - 新增方法级 `drop_last_col` 控制
  - 当前默认配置下已重算一版 `RQ4` 正文主表到：
    - `/Users/chenyufei/Documents/RecSys26/tmp/rq4_sid_geometry_20260424_native/`
- 当前 native-distance 主表结果：
  - `I02-TIGER`: `J@20=0.069283`, `RBO@20=0.101570`
  - `I03-SEATER`: `J@20=0.013739`, `RBO@20=0.018789`
  - `I04-SemID`: `J@20=0.026286`, `RBO@20=0.021477`
  - `I05-LETTER`: `J@20=0.038517`, `RBO@20=0.048360`
  - `I06-DiffGRM`: `J@20=0.172434`, `RBO@20=0.180635`
  - `I07-RQKmeans`: `J@20=0.128040`, `RBO@20=0.196671`
- `RQ4` case study 当前已切到新的 `styled local-scatter` 版本：
  - 已查看 `10471 / 792 / 18089 / 9906 / 14885` 等候选，当前最终正文候选 anchor：`18089`
  - 图目录：`/Users/chenyufei/Documents/RecSys26/tmp/rq4_sid_geometry_20260424_case_18089_muted_topk_relaxed_compact_v10/`
  - 可视化修正：
    - 对齐 DECOR-style 的低饱和配色语义和 marker 设计
    - 去掉 anchor-to-SID 连线，只保留真实散点分布
    - 改为更紧凑的 `2x3` panel 布局，调整整图宽高比以匹配方形 panel，增大子图占比并压缩行列间距
    - 去掉参考圆环，避免视觉上像同心圆示意图
    - 视窗只保证 `anchor / reference top-K / SID top-K` 完整可见，不让远处灰色 context outlier 压缩主体分布
    - 只对红/蓝/紫 top-K marker 做轻量 deterministic 防重叠显示处理，灰色 context 保持原始投影位置
    - 提高灰色 context 可见度，并对视窗外 context 做过滤，避免 marker 贴边时产生越框假象
  - 选择 `18089` 的原因：reference top-K 和 SID top-K 在二维投影中更展开，`I06/I07` 的 overlap 仍然清楚，同时不会像 `9906` 那样大量 marker 聚在同一区域。

### 2026-03-02

- 初始化本地追踪文件：`Process.md`、`tasks.csv`（本机为唯一真源）。
- 核验远端新数据集 splits：`/data/xqp_data/RecSys26/third_party/SETRec/data/{amazon23_vg,microlens_50k,yelp}/` 已包含 `training_dict.npy` 等必需文件；clean SETRec 已建立 symlink 方便训练入口复用。
- `microlens_50k`：已生成 `combine_tdcb_maps.npy`（来自 `MicroLens-50k_titles.csv`；覆盖 19162 items，无缺失）。
- `microlens_50k`：已生成语义 embedding `microlens_50k.emb-t5-tdcb.npy`（`sentence-transformers/sentence-t5-base`；shape=(19162,768)，float32；显式使用 `CUDA_VISIBLE_DEVICES=3`，避开 GPU2）。
- `amazon23_vg`：已生成 `combine_tdcb_maps.npy`（对齐 `item_map_reverse.npy` + `meta_Video_Games.jsonl.gz`；覆盖 96462 items，无缺失）。
- `yelp`：已生成 `combine_tdcb_maps.npy`（对齐 `item_map_reverse.npy` + `yelp_academic_dataset_business.json`；覆盖 149124 items，无缺失）。
- `microlens_50k`：已生成 SEATER/SASRec 训练所需 TSV：`/data/xqp_data/RecSys26/datasets/seater_setrec/microlens_50k/dataset/{training,validation,test}.tsv`（item +1 shift）。
- SEATER 适配：已为新数据集加入 `SETRec_microlens_50k` 配置（`third_party/SEATER/config/const.py`、`third_party/SEATER/main.py`、`third_party/SEATER/config/SETRec_microlens_50k/`）。
- `microlens_50k`：SASRec 训练完成（run：`/data/xqp_data/RecSys26/logs/sasrec_embed/microlens_50k/SASREC_SETRec_microlens_50k__3_2_9`），并已从 `ckpt/best.pth` 导出 `SASRec_item_embed.pkl`：`/data/xqp_data/RecSys26/third_party/SETRec/data/microlens_50k/SASRec_item_embed.pkl`（shape=(19162,64)，torch.float32）。
- `microlens_50k`：已运行 `scripts/prepare_setrec_adapters.py --domains microlens_50k --overwrite`，生成下游方法所需的适配数据到 `datasets/{eager_setrec,etegrec_setrec,seater_setrec,rqvae_recommender}/microlens_50k/`。
- 基础设施：已修补 `scripts/unified_eval.py` 的 `--dataset` 白名单，新增支持 `amazon23_vg`/`microlens_50k`/`yelp`，以便对新数据集统一计算 `R@5`/`R@10`/`N@5`/`N@10` 等指标。
- 基础设施：已扩展远端复现入口脚本的 domain 白名单（`scripts/{repro_dispatch_paper.sh,repro_dispatch.sh,launch_repro_domain.sh,from_scratch/*}`），新增支持 `amazon23_vg`/`microlens_50k`/`yelp`（已做 `bash -n` 语法校验）。
- 基础设施：已修复 `scripts/from_scratch/repro_setrec_t5.sh` 的 `DATA_PATH` 为 `../data/<domain>/`，避免 SETRec `finetune_t5.py` 内部对 `../data/{args.data_path}/...` 的路径拼接导致绝对路径传参失效。
- `microlens_50k`：已启动 SETRec（paper profile）训练→导出→统一评测流水线：`/data/xqp_data/RecSys26/logs/repro_from_scratch/setrec/microlens_50k/20260302_133330`（`CUDA_VISIBLE_DEVICES=3,4,5,6`；日志：`run.log`；完成后产出 `pred_topk.jsonl` 与 `metrics_full.json` 并同步到 `pred_exports/setrec/`）。
- 基础设施：已清理脚本中的 GPU2 默认值（例如 `scripts/build_letter_indices_setrec.sh` 默认不再指向 GPU2；`queue_*` 脚本默认 GPU pool 去除 2），确保“默认行为”也符合避开 GPU2 的约束。
- `amazon23_vg`：已生成 SEATER/SASRec 训练所需 TSV：`/data/xqp_data/RecSys26/datasets/seater_setrec/amazon23_vg/dataset/{training,validation,test}.tsv`（his_seq 截断到 50；train_rows=1279554，val_rows=41203，test_rows=41112；item 已 +1 shift）。
- `yelp`：已生成 SEATER/SASRec 训练所需 TSV：`/data/xqp_data/RecSys26/datasets/seater_setrec/yelp/dataset/{training,validation,test}.tsv`（his_seq 截断到 50；train_rows=3294841，val_rows=120488，test_rows=106511；item 已 +1 shift）。
- SEATER 适配：已为新数据集加入 `SETRec_amazon23_vg` 与 `SETRec_yelp` 配置（`third_party/SEATER/config/const.py`、`third_party/SEATER/main.py`、`third_party/SEATER/config/SETRec_amazon23_vg/`、`third_party/SEATER/config/SETRec_yelp/`）；并创建 `datasets/seater_setrec/{amazon23_vg,yelp}/{tree_data_SASREC,vocab}/` 目录供后续导出与树构建使用。
- EAGER 适配：已将 `patches/from_scratch_20260228/EAGER.patch` 应用到 `third_party_clean/EAGER/`，补齐 `EAGER/train_rec_setrec.py` 并扩展 `--domain` choices 支持 `amazon23_vg`/`microlens_50k`/`yelp`（该脚本依赖 `SASRec_item_embed.pkl` + `*.emb-t5-tdcb.npy`，需先完成对应资产）。 
- SEATER 适配（clean）：已将 `patches/from_scratch_20260228/SEATER.patch` 应用到 `third_party_clean/SEATER/`，并补齐 `SETRec_microlens_50k`/`SETRec_amazon23_vg`/`SETRec_yelp` 的 config/YAML，使 `scripts/from_scratch/repro_seater.sh` 可直接使用 clean 代码跑新域。
- DiffGRM 适配（clean）：已将 `patches/from_scratch_20260228/DiffGRM.patch` 应用到 `third_party_clean/DiffGRM/`，并扩展 SETRec `available_categories` 支持 `amazon23_vg`/`microlens_50k`/`yelp`（依赖 `combine_tdcb_maps.npy` 与 `*.emb-t5-tdcb.npy`）。
- `microlens_50k`：SETRec（paper profile）已完成 `finetune_t5.py`；当前在 `run_dir=/data/xqp_data/RecSys26/logs/repro_from_scratch/setrec/microlens_50k/20260302_133330` 执行 `export_setrec_t5_topk.py`（beta 选择 + test 导出，单卡 GPU3；pid=473614）；`pred_topk.jsonl` 已开始生成（当时 `wc -l`=3566），待完成后运行 `unified_eval.py` 并同步到 `pred_exports/setrec/`。
- 更新：`microlens_50k`：SETRec（paper profile）**全流程已完成**（train→export→unified_eval），run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/setrec/microlens_50k/20260302_133330`；导出与评测产物已同步到：
  - `pred_exports/setrec/microlens_50k_pred_topk.jsonl`
  - `pred_exports/setrec/microlens_50k_pred_topk_metrics_full.json`
  - `logs/repro_from_scratch/setrec/microlens_50k/20260302_133330/run_manifest.json`
- `microlens_50k`：统一评测（`unified_eval.py --mode full`，All performance / 全候选集）主指标：
  - `R@5=0.0025; R@10=0.0040; N@5=0.0019; N@10=0.0025`
- 说明（关键）：`microlens_50k` 测试集中 **cold item 交互占比约 81.9%（39254/47912）**。SETRec 在“全候选集 topK（All performance）”下 topK 几乎全为 warm item，导致包含 cold ground-truth 的总体指标显著偏低。按 SETRec 官方训练脚本 `finetune_t5.py` 的 warm/cold 候选集 mask 口径（beta=0.1；validation 选 beta，避免 test 泄漏），可得到更可解释的拆分指标：
  - Warm（mask cold candidates）：`R@5=0.0133; R@10=0.0220; N@5=0.0090; N@10=0.0120`
  - Cold（mask warm candidates）：`R@5=0.0149; R@10=0.0226; N@5=0.0108; N@10=0.0137`
- 复核：SETRec 的 splits 生成逻辑与官方仓库的预处理 notebook 一致（`third_party_clean/SETRec/data/pre-processing-example/data_preprocess_step1.ipynb`），即：
  - 全局时间戳切分（`split_ratio=0.17`；`split_time1=time_list[-test_num]`；`split_time2=time_list[-ceil(1.8*test_num)]`），并删除 `training_old_dict[u_id] < 2` 的用户；
  - warm item 定义为训练集中出现过的 item；val/test 中未出现在训练的为 cold item；
  - item_map 由 item_set 去重后按 seed=2023 shuffle，再映射为 0-based canonical id。
- 进一步诊断（`microlens_50k` 的 “All 极低”根因定位）：
  - `pred_topk.jsonl`（beta=0.1，validation 选 beta）与 `pred_topk_beta0.2.jsonl`（强制 beta=0.2）中，**top20 预测 item 100% 属于 warm（cold=0）**，导致在 full candidates 口径下 **cold_gt 的 R@5/R@10=0**，从而 All 被大量 cold-only 用户拖到很低。
  - 将 beta 调到 1.0（`pred_topk_beta1.0.jsonl`，只依赖 semantic tokens）后，top20 预测中 cold 占比约 **11.5%**，full candidates 下 cold_gt 的 `R@10≈0.0040` 但 `R@5≈0.0004`，All 的 `R@5` 反而下降（说明该 split 上“冷 item 进入 topK”与“warm 命中”存在强权衡，且 cold 在 full candidates 下仍难以进入 top5）。

- 按你的最新要求：**暂时跳过 SETRec 的进一步复现**，先在同一份 splits 上跑其它方法，检查是否同样出现 “All(full candidates) 异常偏低” 的现象。
- `microlens_50k`：SEATER（smoke profile，epochs=1）已跑通并完成统一评测（run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/seater/microlens_50k/20260302_180210`；产物同步到 `pred_exports/seater/`）。All(full candidates) 主指标：
  - `R@5=0.0007; R@10=0.0017; N@5=0.0005; N@10=0.0008`
  - 该数值同样很低（甚至低于 SETRec 的 All），支持“问题可能主要来自 splits 分布/评测口径与 full candidates 下的冷启动难度”，而非仅某个方法实现 bug。
  - 进一步核验：SEATER 的导出 `pred_topk.jsonl`（top20）中 **cold 预测占比=0.0（498200 条预测里 cold=0；24910/24910 用户均为 0）**，因此在 cold-heavy 的测试集下 All 会被显著拉低。
- `microlens_50k`：SEATER（paper profile，epochs=100）已启动（run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/seater/microlens_50k/20260302_183546`；GPU5；完成后同样用 `unified_eval.py --mode full --top_n 5,10` 记主指标）。
- EAGER：在 `third_party_clean/EAGER` 上发现并修复两类阻塞问题（已在 `tasks.csv:T9008` 记录）：
  - `generate_training_batches.py`：去除硬编码保存到 `/home/.../Amazon_Beauty/`，改为使用 `processed/his_maxtix.pt` 与 `processed/labels.pt` 缓存；
  - `KmeansTree.py`：修复 `choices/index` device mismatch，并将 joblib `Parallel(n_jobs)` 固定为 1 以避免多进程 GPU 聚类不稳定；
  - 下一步：重新跑通 `repro_eager.sh microlens_50k smoke` 后再上 paper profile。
- 更新：EAGER（smoke profile）已重新启动（run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/eager/microlens_50k/20260302_184647`；GPU6），待完成后用 `unified_eval.py` 记录 All(full candidates) 指标并检查是否仍出现 warm-only 预测。
- `microlens_50k`：DiffGRM（smoke profile，epochs=1）已启动（run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/diffgrm/microlens_50k/20260302_183019`；GPU4；包含 OPQ/语义ID 构建、训练、导出与 unified_eval；待完成后记录 All(full candidates) 主指标）。

### 2026-03-02（5-core 对齐 & 重划分）

- 背景：你指出 SETRec 原论文/代码使用 **5-core filtered** 数据（而我们此前在新域上是从 raw 0-core 起步）。为对齐实验假设，已将 `amazon23_vg` / `microlens_50k` / `yelp` 三个新域切换为 **5-core** 口径并重建全部下游资产。

- Amazon23 Video Games：确认 Amazon Reviews'23 官方提供 5-core benchmark（Video_Games 有直链），并已下载 `rating_only/Video_Games.csv.gz`：  
  - 参考：[Amazon Reviews'23 (McAuley Lab), 5-core page, https://amazon-reviews-2023.github.io/data_processing/5core.html]  
  - 下载：[McAuley Lab, Video_Games 5-core rating_only, https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz]  
  - 落盘：`/data/xqp_data/RecSys26/datasets/amazon23_vg/raw/Video_Games_5core_rating_only.csv.gz`  
  - sha256：`a2bde5f3b945960d161538c200dd87845e6ee471b46da96410dde61613c6901c`

- MicroLens-50k / Yelp：未发现官方发布的“已 5-core 过滤 + 与 SETRec 格式直接兼容”的成品包，因此使用 `scripts/make_setrec_splits.py --k_core 5` 自行做迭代 k-core 过滤并重切分。

- split 工具增强：`scripts/make_setrec_splits.py` 新增 `--format amazon23_benchmark_csv`，支持直接读取 Amazon Reviews'23 官方 5-core `*.csv.gz`（并保持“按全局时间戳切分”的 SETRec 风格）。

- 新生成的 k5 splits：
  - `/data/xqp_data/RecSys26/third_party/SETRec/data/{microlens_50k_k5,amazon23_vg_k5,yelp_k5}/`
  - 并将默认域名软链切到 k5（旧 k0 目录备份）：
    - `microlens_50k -> microlens_50k_k5`（backup=`microlens_50k_k0_20260302_193456`）
    - `amazon23_vg -> amazon23_vg_k5`（backup=`amazon23_vg_k0_20260302_193456`）
    - `yelp -> yelp_k5`（backup=`yelp_k0_20260302_193456`）

- k5 splits 统计（来自 manifest + warm/cold dict 计数；cold_frac=测试集中 cold 交互占比）：
  - `microlens_50k`：`n_users=42666, n_items=14079, test_cold_frac=0.8192`
  - `amazon23_vg`：`n_users=74333, n_items=25062, test_cold_frac=0.7526`
  - `yelp`：`n_users=221039, n_items=109326, test_cold_frac=0.2475`
  - 备注：`microlens_50k`/`amazon23_vg` 的 cold_frac 仍然很高，因此 “All(full candidates)” 口径仍可能偏低——后续需进一步对齐“论文中 all 的定义/评测口径”或检查 split 逻辑是否与目标论文一致（此处先记录为待排查点）。

- 为 k5 重新生成 item 文本与语义 embedding（均落盘到 `third_party/SETRec/data/<domain>/`）：
  - 新增脚本并同步到远端：`scripts/{build_setrec_combine_tdcb_maps.py,build_setrec_t5_embeddings.py}`  
  - 结果：
    - `microlens_50k`: `combine_tdcb_maps.npy`(n=14079), `microlens_50k.emb-t5-tdcb.npy`(14079,768)
    - `amazon23_vg`: `combine_tdcb_maps.npy`(n=25062), `amazon23_vg.emb-t5-tdcb.npy`(25062,768)
    - `yelp`: `combine_tdcb_maps.npy`(n=109326), `yelp.emb-t5-tdcb.npy`(109326,768)

- 为 k5 重新生成 SEATER/SASRec TSV（落盘到 `datasets/seater_setrec/<domain>/dataset/*.tsv`）：
  - 新增脚本并同步到远端：`scripts/build_seater_setrec_tsv.py`
  - 结果行数：
    - `microlens_50k`: train=185987, val=21588, test=23169（`wc -l` 含 header）
    - `amazon23_vg`: train=486432, val=22796, test=20633
    - `yelp`: train=2542843, val=92634, test=83449

- 更新 SEATER dataset config（k5 item/user 数）：已修改 `third_party/SEATER/config/const.py` 中 `SETRec_{microlens_50k,amazon23_vg,yelp}` 的 `users_ID_num/item_ID_num` 以匹配 k5。

- k5 的 SASRec（CF tokenizer）：
  - `microlens_50k`：SASRec 训练完成（run：`/data/xqp_data/RecSys26/logs/sasrec_embed_k5/microlens_50k/SASREC_SETRec_microlens_50k_k5_20260302_195827__3_2_19`；GPU4），并导出：  
    - `third_party/SETRec/data/microlens_50k/SASRec_item_embed.pkl`（shape=(14079,64)，torch.float32）  
    - 导出脚本：`scripts/export_sasrec_item_embed.py`
  - `microlens_50k`：已重新运行 `scripts/prepare_setrec_adapters.py --domains microlens_50k --overwrite`，生成下游方法适配数据（k5 口径）。
  - `amazon23_vg`：SASRec 训练完成（run：`/data/xqp_data/RecSys26/logs/sasrec_embed_k5/amazon23_vg/SASREC_SETRec_amazon23_vg_k5_20260302_200142__3_2_20`；GPU4），并导出：  
    - `third_party/SETRec/data/amazon23_vg/SASRec_item_embed.pkl`（shape=(25062,64)，torch.float32）
  - `amazon23_vg`：已运行 `scripts/prepare_setrec_adapters.py --domains amazon23_vg --overwrite`，生成下游方法适配数据（k5 口径）。
  - `yelp`：SASRec 训练已结束（run：`/data/xqp_data/RecSys26/logs/sasrec_embed_k5/yelp/SASREC_SETRec_yelp_k5_20260302_200235__3_2_20`），并已从最终 `ckpt/best.pth` 导出：  
    - `third_party/SETRec/data/yelp/SASRec_item_embed.pkl`（shape=(109326,64)，torch.float32）
  - `yelp`：已重跑 `scripts/prepare_setrec_adapters.py --domains yelp --overwrite`，确保下游方法读取到最新 embeddings（k5 口径）。

- 同步修复（避免 k5 口径下游脚本/复现流水线报错）：
  - `third_party_clean/SEATER/config/const.py`：将 `SETRec_{microlens_50k,amazon23_vg,yelp}` 的 `users_ID_num/item_ID_num` 同步到 k5（与 `third_party/SEATER` 一致）。
  - `scripts/exports/export_diffgrm_topk.py`：扩展 `--domain` choices，新增 `amazon23_vg`/`microlens_50k`/`yelp`，避免 DiffGRM 导出阶段因参数白名单失败。

- ⚠️ 操作事故（已按你的指示修复并持续监控）：
  - 误 kill 的训练为 `SETRec + microlens_50k_care`（run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/setrec/microlens_50k_care/20260302_201347/`）。
  - 已从 `out/checkpoint-4400/` **恢复继续训练**（当前 torchrun 主进程 PID=623355；log：同目录 `run.log`）。
  - 备注：尝试用更少 GPU（`--nproc_per_node=2`）恢复会因 checkpoint 内保存的 CUDA RNG state 与可见 GPU 数不一致而报错，因此恢复阶段必须使用与原训练一致的 GPU 数（此处为 4）。

- 评测口径核验（用于解释 “All(full candidates) 过低” 是否来自口径不一致）：
  - SETRec 官方 `code/inference_t5.py` 的 `All performance`：对 `ratings` 直接 `topk`（**不 mask 候选集**），`gold_list` 为每用户的 `groundTruth` 列表，指标由 `computeTopNAccuracy` 计算（按用户平均，`userHit/len(GT)` 与 `dcg/idcg` 的定义与我们一致）。
  - 结论：`scripts/unified_eval.py` 的 Recall/NDCG 计算逻辑已与 SETRec `computeTopNAccuracy` 对齐；因此 All 异常偏低更可能来自 **数据 split 的冷启动占比过高** 或 **模型预测几乎不含 cold item**，而不是“评测口径写错”。（后续若要对齐论文展示的 all/warm/cold 差距，需要进一步确认论文所用数据分布与 split 策略是否与当前新域一致。）

- `microlens_50k_care`：SETRec 训练与预测导出已完成（run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/setrec/microlens_50k_care/20260302_201347/`），产物：
  - `out/pred_topk.jsonl`（生成时间约 2026-03-02 22:52，run.log：`=== End. Best beta is 0.1`）
  - `metrics_full.json`：已执行 `scripts/unified_eval.py --mode full --top_n 5,10`（PBT checks: ALL PASS），主指标：
    - `R@5=0.0411; R@10=0.0652; N@5=0.0261; N@10=0.0343`

- k5 paper profile 复现继续推进（均使用 `with_gpu_lock.sh`，GPU pool=`3 4 5 6`，避开 GPU2，且总并发 GPU 数 ≤ 4）：
  - 已启动 `SEATER + microlens_50k`（paper，epochs=100）：`/data/xqp_data/RecSys26/logs/repro_from_scratch/seater/microlens_50k/20260302_230124/`（PID=650019）
  - 已启动 `EAGER + microlens_50k`（paper）：`/data/xqp_data/RecSys26/logs/repro_from_scratch/eager/microlens_50k/20260302_230204/`（PID=650733）
  - 已启动 `DiffGRM + microlens_50k`（paper）：`/data/xqp_data/RecSys26/logs/repro_from_scratch/diffgrm/microlens_50k/20260302_230222/`（PID=651025）

- ETEGRec（SIGIR'25）适配新域配置：已新增
  - `third_party/ETEGRec/config/setrec_microlens_50k.yaml`
  - `third_party/ETEGRec/config/setrec_amazon23_vg.yaml`
  - `third_party/ETEGRec/config/setrec_yelp.yaml`
  - 说明：三份配置均指向 `datasets/etegrec_setrec/<domain>/` 下的 `<domain>.emb_map.json` 与 `<domain>_emb_768.npy`（由 `prepare_setrec_adapters.py` 生成/链接）。

- 任务清单扩展：`tasks.csv` 已补充 `ETEGRec/TIGER/RPG/LLM_RecSys_ID(HowToIndex variants)/LETTER` 在 `amazon23_vg/microlens_50k/yelp` 三域上的 paper-profile TODO（LETTER 暂因缺少 `.index.json` 生成流程而 BLOCKED，需先查明官方做法并实现）。

- `microlens_50k`（k5，paper profile）阶段性结果与问题定位：
  - `EAGER`：旧 run 已完整跑通 train→export→`unified_eval(mode=full, top_n=5,10)`（run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/eager/microlens_50k/20260302_230204/`），但主指标异常偏低：
    - `R@5=0.0011; R@10=0.0016; N@5=0.0008; N@10=0.0010`
  - `EAGER` 低指标根因 review（已修复并在 paper profile 上 rerun）：
    - **样本生成二次 holdout**：`EAGER/lib/generate_train_and_test_data.py:_partial_gen_train_sample` 内部固定 `-2`（预留 val/test），但 SETRec 数据已提供 `training_dict.npy/validation_dict.npy/testing_dict.npy`，导致再 holdout 会让大量短序列用户（Microlens 平均 train 长度 ~5.x）**几乎无训练样本**；修复：为 `_gen_train_sample` 增加 `holdout_last_n`（默认=2 保持原行为），并在 `EAGER/train_rec_setrec.py` 显式设为 `holdout_last_n=0`。该修复后 `microlens_50k` 训练 instances 数从 **35210→72400**。
    - **padding id 不一致**：`Train_instance` 会将 `-1` padding 映射为 `item_num`，但原 `HF_Model.src_pad = src_voc_size - 1`（约 `2*item_num`），导致 attention mask 无法正确 mask padding；修复：将 `src_pad` 对齐到 `item_num`（见 `EAGER/lib/HF_Model.py`）。
    - **pad path 长度假设**：`EAGER/lib/Trm4Rec_trainer.py` 里 padding 的 path 写死长度=2，导致当 `tree_height!=2`（例如 smoke 使用较小 k）直接报错；修复：padding path 改为按 `tree_height` 自动填充。
    - **额外工程对齐**：为降低 smoke 调试成本，已将 `scripts/from_scratch/repro_eager.sh` 的 smoke `k` 调整为 `128`（避免 `k=64` 导致 `tree_height=3` 和极慢的 262144 leaf code 构造）。
  - `EAGER`（paper，k5）已基于上述修复重新启动 rerun：`/data/xqp_data/RecSys26/logs/repro_from_scratch/eager/microlens_50k/20260303_042556/`（监控中，结束后以 `metrics_full.json` 的 `R@5/R@10/N@5/N@10` 为准回填）。
  - `SEATER`：训练结束后在 `trainer.test()` 阶段报错（run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/seater/microlens_50k/20260302_230124/`）：
    - 错误：`FileNotFoundError: .../ckpt/best.pth`（validate 指标未触发 `save_ckpt()`，导致 test 加载 best 失败）
    - 处理：需要修补 `third_party_clean/SEATER` 的 ckpt 保存条件（至少保证 epoch0 保存一次；并避免因指标取整导致永远不保存），然后重跑该任务。
  - `DiffGRM`：导出阶段失败（run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/diffgrm/microlens_50k/20260302_230222/`）：
    - 错误：`user coverage mismatch ... split_users=24910 gt_users=23169`
    - 判断：大概率为 DiffGRM `cache_dir` 复用导致读取了旧 split（k0）缓存的 `processed/all_item_seqs.json`；需清理/更换 cache_dir（k5 tag）后重跑。

### 2026-03-03

- `microlens_50k`（k5 split 的 cold 占比量化，解释 “All 过低” 的关键证据）：
  - 以 `training_dict.npy` 的 item 集合定义 warm item（SETRec 的 warm/cold 定义），则：
    - `train_items=11346`
    - `test_interactions=44430` 中 `cold_interactions=36399`（cold interaction 占比 **0.819**）
    - `test_users=42666` 中至少包含 cold test item 的用户 `cold_users=20855`（cold user 占比 **0.489**）
  - 结论：在该 split 下，若方法**无法推荐训练未出现的 item（cold）**，则 `All(full)` 会被大量 cold case 拉低；warm/cold 拆分看起来“更正常/差距更大”主要由数据分布驱动，并非评测脚本本身错误。

- k5 splits 的跨域 cold 统计（直接验证“5-core 过滤解决不了时间分布偏移”）：
  - 口径：warm=出现在 `training_dict.npy` 的 item；cold=未出现在训练但出现在 val/test 的 item（与 SETRec 产物 `warm_item.npy/cold_item.npy` 一致）。
  - `amazon23_vg`（5-core rating_only + global time split, split_ratio=0.17）：
    - `cold_item_frac=0.284`（cold items / 全部 items）
    - `test_unique_cold_frac=0.615`（test 出现过的 unique items 中 cold 占比）
    - `test_cold_inter_frac=0.753`（test 交互中 cold 交互占比）
  - `microlens_50k`：
    - `cold_item_frac=0.194`
    - `test_unique_cold_frac=0.443`
    - `test_cold_inter_frac=0.819`
  - `yelp`：
    - `cold_item_frac=0.085`
    - `test_unique_cold_frac=0.131`
    - `test_cold_inter_frac=0.247`
  - 结论：`amazon23_vg/microlens_50k` 的 `All(full)` 被 cold case 显著主导；仅做 5-core 并不能约束“交互是否在训练期出现”，因此无法消除该时间分布偏移。

- `SEATER + microlens_50k`（k5）进一步排障与推进：
  - 发现并修复 tree 资产不一致（会导致 export 崩溃/指标异常）：
    - 现象：`datasets/seater_setrec/microlens_50k/tree_data_SASREC/8_branch_tree/itemID_2_tree_indexID.npy` 形状曾为 `(19163,6)`，与 k5 的 `n_items+1=14080` 不一致，导出时报 `item id out of range after shift`。
    - 处理：删除该 `8_branch_tree/`，由 SEATER 在新口径下重建（已在 smoke run `20260303_045424` 验证重建成功，且 `unified_eval` 的 item range check 通过）。
  - smoke 复核（`/data/xqp_data/RecSys26/logs/repro_from_scratch/seater/microlens_50k/20260303_045424/`）：
    - `All(full)`：`R@5=0.0011; R@10=0.0020; N@5=0.0009; N@10=0.0012`
    - warm/cold 分解（用 `testing_{warm,cold}_dict.npy` 重新计算）：
      - warm：`R@5=0.0058; R@10=0.0099; N@5=0.0040; N@10=0.0053`
      - cold：`R@5=0.0000; R@10=0.0000; N@5=0.0000; N@10=0.0000`
    - 解释：该 smoke（epochs=1）下模型几乎不命中 cold item，因此 All 被冷启动样本显著拉低。
  - paper run（epochs=100）已完成：`/data/xqp_data/RecSys26/logs/repro_from_scratch/seater/microlens_50k/20260303_050559/`（`unified_eval` strict + PBT PASS；`All(full)`：`R@5=0.0042; R@10=0.0066; N@5=0.0031; N@10=0.0040`）。
  - 2026-03-07 复核：基于同一 `pred_topk.jsonl` 重跑 `warm_only(strict)`，得到 `R@5=0.0221; R@10=0.0356; N@5=0.0143; N@10=0.0189`；保留用户 `6482/23169`（28.0%）；与已有 `metrics_warm_only.json` 一致。

- `EAGER + microlens_50k`（k5 paper rerun）已完成：
  - run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/eager/microlens_50k/20260303_042556/`（DONE：2026-03-03 05:32:56 UTC）
  - `All(full)`：`R@5=0.0029; R@10=0.0046; N@5=0.0022; N@10=0.0029`（PBT PASS）

- `DiffGRM + microlens_50k`（k5）cache_dir 复用问题修复与 smoke 验证（已完成）：
  - 修复：将 `scripts/from_scratch/repro_diffgrm.sh` 的 `CACHE_DIR` 改为按域隔离的 `/data/xqp_data/RecSys26/datasets/diffgrm_cache_from_scratch/${domain}_k5`，避免 k0/k5 split 混用导致 `user coverage mismatch`。
  - smoke run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/diffgrm/microlens_50k/20260303_051509/`（DONE：2026-03-03 05:30:28 UTC；PBT PASS）
  - smoke `All(full)`：`R@5=0.0004; R@10=0.0007; N@5=0.0003; N@10=0.0004`

- 复现策略调整（基于上述 split 结论）：
  - 短期：优先在 `yelp` 上完成各方法的 paper-profile 复现与指标回填，用于验证“正常量级”的 `R@5/R@10/N@5/N@10`。
  - `amazon23_vg/microlens_50k`：保留当前 k5 split 的复现作为 **cold stress test**，但不再将 `All(full)` 直接与论文 all 列对齐；必要时需另行设计“时间感知的 warm-only split”或其它对齐策略再做严格对比。
  - 已启动 `yelp` 的 sanity runs（用于尽快验证 all 的量级是否恢复正常）：
    - `SETRec + yelp (smoke)`：`/data/xqp_data/RecSys26/logs/repro_from_scratch/setrec/yelp/20260303_052428/`（原先在 GPU6；按你的指示已停止并迁移）
    - `SEATER + yelp (smoke)`：`/data/xqp_data/RecSys26/logs/repro_from_scratch/seater/yelp/20260303_053102/`（CUDA_VISIBLE_DEVICES=4；同时构建 `8_branch_tree`）
  - `SETRec + yelp (smoke)` 迁移执行（避免 GPU6，改用 `GPU1/3/4/5` 中的空闲卡；本次锁到 GPU1）：
    - 已向 `torchrun` 主进程发送 `SIGTERM` 停止 GPU6 上的长时间 test beta sweep，保留 best checkpoint：`.../out/checkpoint-8000/`。
    - 修复 `export_setrec_t5_topk.py` 的数据前缀 bug：此前 `Path.resolve()` 会把 `.../data/yelp -> .../yelp_k5` 导致错误寻找 `yelp_k5.emb-t5-tdcb.npy`；现改为从 `args.data_path` 取前缀，适配 `*_k5` 真实目录名。
    - 使用 checkpoint-8000 “finalize” 生成 `out/adapter.pth` 与保存的 T5 权重后，已完成 `export_setrec_t5_topk.py` + `unified_eval`（详见：`/data/xqp_data/RecSys26/logs/repro_from_scratch/setrec/yelp/20260303_052428/resume_export_eval.log`）。
  - `SETRec + yelp (smoke)` 已完成（DONE=2026-03-03T07:33:43Z）：
    - run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/setrec/yelp/20260303_052428/`（实际 export+eval 使用 GPU1）
    - beta（val 选取）：`0.5`（`val Recall@5=0.015`）
    - `All(full)`：`R@5=0.0091; R@10=0.0164; N@5=0.0112; N@10=0.0132`（`unified_eval` strict + PBT PASS）

- `SETRec + yelp (paper)` 已完成：
  - run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/setrec/yelp/20260303_074456/`（GPU=`1,3,4,5`；避开 GPU2/GPU6；最多占用 4 张 GPU）
  - beta：`0.3`（脚本内置 sweep 的 best beta；为节省时间未做全量 val sweep）
  - beta sweep 详情：`/data/xqp_data/RecSys26/logs/repro_from_scratch/setrec/yelp/20260303_074456/beta_results_live.tsv`（best beta=0.3；`beta=1.0` 与其同分）
  - `unified_eval(mode=full, strict, top_n=5,10)`：`R@5=0.0119; R@10=0.0210; N@5=0.0137; N@10=0.0164`
  - 产物：`pred_topk.jsonl`、`metrics_full.json`、`beta.json`、`run_manifest.json`
  - pred_exports 同步：
    - `/data/xqp_data/RecSys26/pred_exports/setrec/yelp_pred_topk.jsonl`
    - `/data/xqp_data/RecSys26/pred_exports/setrec/yelp_pred_topk_metrics_full.json`（原 smoke 备份为 `yelp_smoke_20260303_052428_*`）

- `SEATER + yelp (smoke)` 已完成（DONE=2026-03-03T06:03:04Z）：
  - run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/seater/yelp/20260303_053102/`
  - `All(full)`：`R@5=0.0083; R@10=0.0147; N@5=0.0101; N@10=0.0120`（`unified_eval` strict + PBT PASS）
  - warm/cold 分解（用 `testing_{warm,cold}_dict.npy` 重新计算）：
    - warm：`R@5=0.0112; R@10=0.0196; N@5=0.0120; N@10=0.0147`
    - cold：`R@5=0.0000; R@10=0.0000; N@5=0.0000; N@10=0.0000`
  - 解释：在该口径下 SEATER 仍几乎无法命中“训练期未出现的 item（cold）”；但 yelp 的 `test_cold_inter_frac` 仅 `0.247`，因此 `All(full)` 的量级较 `amazon23_vg/microlens_50k` 明显提升。

- `yelp` 的 paper-profile 复现进度（GPU pool=`0 1 3 4`；避开 GPU2/GPU5/GPU6；并发 GPU 数 ≤ 4）：
  - `SEATER + yelp (paper)` 已完成（DONE=2026-03-03T19:20:30Z）：
    - run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/seater/yelp/20260303_162658/`
    - `unified_eval(mode=full, strict, top_n=5,10)`：`R@5=0.0115; R@10=0.0205; N@5=0.0136; N@10=0.0164`（PBT PASS）
    - exports：`/data/xqp_data/RecSys26/pred_exports/seater/yelp_pred_topk_metrics_full.json`
  - `EAGER + yelp (paper)` 已完成（DONE=2026-03-03T20:50:32Z；PBT PASS）：
    - run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/eager/yelp/20260303_162656/`
    - `unified_eval(mode=full, strict, top_n=5,10)`：`R@5=0.0060; R@10=0.0101; N@5=0.0069; N@10=0.0081`
    - exports：`/data/xqp_data/RecSys26/pred_exports/eager/yelp_pred_topk_metrics_full.json`
    - 进一步排查（为什么比 SETRec/SEATER 低）：
      - warm/cold 拆分（使用 `testing_warm_dict.npy`/`testing_cold_dict.npy` 重算，与 SEATER/SETRec 一致口径）：
        - EAGER warm：`R@5=0.0079; R@10=0.0133; N@5=0.0082; N@10=0.0098`
        - EAGER cold：`R@5=0.0000; R@10=0.0000; N@5=0.0000; N@10=0.0000`
        - 对比（同一口径）：SETRec warm `R@5≈0.0157`，SEATER warm `R@5≈0.0154`；三者 cold 均为 0。
      - 关键不对齐点（高优先级怀疑根因）：EAGER 原仓库训练流程要求先训练 DIN（行为流 embedding），再训练 EAGER。
        - 原代码（`third_party_clean/EAGER/EAGER/train_rec.py:228`）明确加载 `DIN_Model` 并取 `data1=DIN_Model.item_embedding...`（行为流，维度=96），`data2` 为 T5 语义 embedding（维度=768）。
        - 当前 RecSys26 的 SETRec 适配脚本（`third_party_clean/EAGER/EAGER/train_rec_setrec.py:369`）行为流改为读取 `SASRec_item_embed.pkl`（维度=64），**未按论文/README 的 DIN 预训练流程**，因此性能偏低很可能来自行为流建树与表征不对齐（且 embedding 维度与论文不同）。
      - 结论：当前这次 EAGER(yelp) 结果属于“可跑通 + 评测正确（PBT PASS）”，但并非严格 paper-aligned；若要严格对齐，需要补上 DIN 预训练并用 DIN embedding 作为行为流输入，再重跑 EAGER。
  - `DiffGRM + yelp (paper)` 已完成（DONE=2026-03-04T01:49:50Z；PBT PASS）：
    - run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/diffgrm/yelp/20260303_162656/`
    - `unified_eval(mode=full, strict, top_n=5,10)`：`R@5=0.0071; R@10=0.0122; N@5=0.0085; N@10=0.0099`
    - exports：`/data/xqp_data/RecSys26/pred_exports/diffgrm/yelp_pred_topk_metrics_full.json`
  - `ETEGRec + yelp (paper)`：
    - 旧 run（失败）：`/data/xqp_data/RecSys26/logs/repro_paper/etegrec/yelp/20260303_165140/`（exit_code=1；`ValueError: [TOKENIZER] ... maximum conflict: 9309 > 4096`；`map_path` 拼接错误已修复为 `.emb_map.json`）
    - 新 run（按官方流程先做 tokenizer 预训练）：`/data/xqp_data/RecSys26/logs/repro_paper/etegrec_rqvae/yelp/20260303_195835/`（tmux=`recsys26_T3013_etegrec_yelp_rqvae_gpu0`；CUDA_VISIBLE_DEVICES=0）
      - 阶段 1：`RQVAE` tokenizer pretrain（epochs=2500；按 paper Scientific 的 `10000*25848/109327` 近似对齐总 steps）
      - 阶段 2：导出 `rqvae_state_dict.pth` 并以 `--rqvae_path` 加载后训练/评测 ETEGRec
      - 监控：`/data/xqp_data/RecSys26/logs/repro_paper/etegrec_rqvae/yelp/latest/run.log`

### 2026-03-04

- 远端 SSH 连接波动（本地→`qipeng@target-server`）：
  - 现象：曾出现 `ssh: ... Operation not permitted` / `Connection closed by UNKNOWN port 65535`（约 `2026-03-04 04:28–04:34`，本地时区）。
  - 现状：`2026-03-03 21:33 UTC` 已恢复，可正常查看 GPU/日志并继续推进。

- EAGER（KDD'24）`yelp`（k5，paper-aligned）启动重跑：补齐 DIN 行为流预训练（严格对齐 README 流程）
  - 代码改动（远端）：
    - 新增：`/data/xqp_data/RecSys26/third_party_clean/EAGER/EAGER/train_din_setrec.py`（在 SETRec splits 上训练 DIN，并保存 `DIN_MODEL_29000.pt`）
    - 修改：`/data/xqp_data/RecSys26/third_party_clean/EAGER/EAGER/train_rec_setrec.py`（新增 `--behavior_emb {sasrec,din}` 与 `--din_model_path`；支持从 DIN checkpoint 读取行为流 item embedding）
    - 修改：`/data/xqp_data/RecSys26/scripts/from_scratch/repro_eager.sh`（paper profile 自动先跑 DIN，再跑 EAGER；并将两步拼入 `train_cmd_str` 写入 manifest）
  - 第一次尝试 run：`/data/xqp_data/RecSys26/logs/repro_from_scratch/eager/yelp/20260304_050216/`
    - DIN pretrain 已完成（`total_batch_num=29000`；产物：`/data/xqp_data/RecSys26/datasets/setrec_data/yelp/DIN_MODEL_29000.pt` + `DIN_MODEL.pt`）
    - 但随后 EAGER 启动因 `train_rec_setrec.py` 内新增分支的 f-string 拼接 bug 报 `SyntaxError: f-string expecting '}'` 而中止；已修复并重启（见下）
  - 当前 run（GPU3）：`/data/xqp_data/RecSys26/logs/repro_from_scratch/eager/yelp/20260304_051437/`
    - 状态：已结束且未产出 `pred_topk.jsonl/metrics_full.json`，`run_manifest.json` 记录 `exit_code=120`；`ckpt/` 为空（训练未开始）
    - run.log 最后停在 `embkm` kmeans tree 的 mini-batch 阶段（持续 `Minbatch...`），仅生成 tree0 映射：`processed/item_to_code_tree0_k128.npy`、`processed/code_to_item_tree0_k128.npy`（tree1 缺失）
    - 下一步：改用 tmux 启动并在同一 run_dir 上续跑（复用 `processed/` 与 tree0 映射），优先把 tree1 生成完成后再进入训练/导出/`unified_eval`

- TIGER（NeurIPS'23）`RQ_VAE_Recommender`：`yelp`（k5，paper profile）当前阻塞：
  - `latest->/data/xqp_data/RecSys26/logs/repro_paper/tiger/yelp/20260303_211802/`，`exit_code=1`
  - 报错：CUDA `device-side assert` / `Indexing.cu: indexSelectLargeIndex`（异步报错；已定位到 semantic id embedding 越界）
  - 初步根因（需修补后重跑验证）：`SemanticIdTokenizer.precompute_corpus_ids` 追加的 dedup 列为“重复次数 hits”，其取值未限制到 `<vae_codebook_size(=256)`，decoder 侧 `SemIdEmbedder` 用 `token_type*256 + sem_id` 做 embedding 索引，导致越界触发 assert。
  - 已做修补（工程偏离点，需记录）：在 `third_party/RQ_VAE_Recommender/modules/tokenizer/semids.py` 的 `precompute_corpus_ids` 中对 `hits` 增加 `clamp_max(codebook_size-1)`，保证 dedup 列不会导致 decoder embedding 越界。影响：若同一 semantic code 的重复次数 >256，则 dedup 编码会发生截断冲突；预期在 paper 级别充分训练的 RQ-VAE 下该情况极少，但 smoke/debug 场景可显著降低 crash 风险。

- DiffGRM（yelp，paper）：已完成（DONE=2026-03-04T01:49:50Z；见上文 yelp paper-profile 列表与 `tasks.csv:T2304`）。

- ETEGRec（yelp）现状（对齐核查）：
  - T3013（GPU0，当前 run 仅作 smoke 参考）：
    - `latest->20260303_195835`；当前到 `Epoch=99`（train 中；每 epoch eval；early_stop=50）
    - 关键问题：该 run 使用 `third_party/ETEGRec/config/setrec_yelp.yaml` 的 smoke 超参（`max_length=64`、`encoder_layers=1/decoder_layers=1`、`warmup_steps=0`、`num_beams=10`、`weight_decay=0`、`e_dim=32` 等），**并非**官方 `scientific.yaml` 的 paper-aligned 设置；因此 Val 极低是预期现象，不能直接当作 paper 结论。
    - 最近一次 Val（`Epoch=98`，官方脚本口径）：`recall@5=0.002494; recall@10=0.003098; ndcg@5=0.001720; ndcg@10=0.001915`
  - T3014（GPU4，paper-aligned config）：
    - 启动时间：`2026-03-04T08:07:35Z`；run：`/data/xqp_data/RecSys26/logs/repro_paper/etegrec_rqvae_papercfg/yelp/20260304_080735/`（latest 同目录）
    - config：`third_party/ETEGRec/config/setrec_yelp_paper.yaml`（以 `third_party_clean/ETEGRec/config/scientific.yaml` 为基准对齐：`max_length=210`、`6-layer`、`d_model=128`、`code_num=256`、`num_beams=20`、`warmup_steps=8000`、`weight_decay=0.05`、`warm_epoch=10`；仅把 `data_path/semantic_hidden_size/semantic_emb_path` 按 yelp 适配）
    - 当前阶段：RQVAE tokenizer pretrain（`e_dim=128`、`layers=512-256`、epochs=2500；按 Scientific 的 10000 epochs 近似对齐总 steps）

### 2026-03-05

- 按用户要求停止 ETEGRec（yelp）两个训练进程（SIGTERM，`2026-03-04T17:06Z`；本地 `2026-03-05T01:06+08:00`）并释放 GPU0/GPU4：
  - T3013（GPU0，smoke config 非 paper-aligned）：`run.log` 到 `Epoch 183`；Val(last epoch183)：`R@5=0.003616; R@10=0.005430; N@5=0.002503; N@10=0.003084`；Val(best N@10 epoch159)：`R@5=0.003865; R@10=0.005603; N@5=0.002689; N@10=0.003252`；ckpt 最后保存 epoch159：`/data/xqp_data/RecSys26/third_party/ETEGRec/myckpt/yelp/Mar-03-2026_19-58-8932de/159.pt`
  - T3014（GPU4，paper-aligned config）：`run.log` 到 `Epoch 39`；Val(last epoch39)：`R@5=0.008679; R@10=0.015048; N@5=0.005416; N@10=0.007449`；Val(best N@10 epoch33)：`R@5=0.009176; R@10=0.016031; N@5=0.005736; N@10=0.007931`；best ckpt epoch33：`/data/xqp_data/RecSys26/third_party/ETEGRec/myckpt/yelp/Mar-04-2026_09-05-e5df20/33.pt`

### 2026-04-14

- 接手 `amazon23_vg` 的 scaling law 实验并完成远端核验：
  - 远端根目录、脚本路径、`logs/scaling_amz23vg/`、`/tmp/vg_{launch,eval,status}.sh` 与交接文档一致。
  - 当前 tmux 会话实际为：`vg_l_I03`（`I03-seater` `t5-large` 训练中）、`sw_I07s`（`I07-rkmeans` `t5-small` sweep 中）；未发现此前摘要里提到的 `sw_I07l_c` 活跃会话。
  - `vg_l_I03` 当前在 `~19.27k/20k` steps；`sw_I07s` 的启动命令已确认是顺序评测 `9999 -> 14999 -> 19999`，当前仍在导出 `9999` 的预测。

- 已复核的 VG cached eval（`/tmp/sweep_vg_*/*/metrics.json`）与 diversity：
  - `I02-semrq @4999`：`t5-small/base/large` 的 `N@10` 分别为 `0.0192 / 0.0212 / 0.0168`；`unique_items=114 / 46 / 16`，mode collapse 结论成立。
  - `I03-seater @4999`：`t5-small/base` 的 `N@10=0.0112 / 0.0093`；`unique_items=2726 / 6017`，当前未见类似 I02 的 collapse。
  - `I07-rkmeans @4999`：`t5-small/base/large` 的 `N@10=0.0196 / 0.0135 / 0.0156`；`unique_items=575 / 4627 / 2571`。已缓存的后续 checkpoint 结果显示：`t5-base` 在 `9999/14999/19999` 分别降到 `0.0092/0.0082/0.0091`；`t5-large` 在 `13999~19999` 区间维持 `0.0075~0.0084`，当前 best 仍是 `4999`。
  - 注意：`I04/I05/I06` 目前目录中仅存在 `M01-t5s/checkpoint_4999.pt`，且对应指标与旧表中的 `wd=0.035` baseline 一致；尚未看到新配置（`lr=1e-3, wd=1e-4`）下的 base/large 运行产物。

- 发现一处配置偏差，需在后续解释结果时显式注明：
  - `scaling_amz23vg_t5base*.gin` 与 `t5large*.gin` 均已设置 `train.cosine_decay=True`；
  - 但当前 `scaling_amz23vg_t5small*.gin`（普通 / grid / seater）均**未**设置 `train.cosine_decay=True`，因此现有 small 组并非“all sizes unified cosine decay”。

- 为补齐缺失矩阵，已启动一个新的 clean run：
  - `I06-diffgrm` `t5-base` 已于 GPU6 启动：tmux=`vg_b_I06`
  - 命令：`bash /tmp/vg_launch.sh I06-diffgrm M02-t5b 6 /data/xqp_data/RecSys26/third_party/RQ_VAE_Recommender/configs/scaling_amz23vg_t5base.gin`
  - 日志：`/tmp/vg_b_I06.log`
  - 当前状态：已正常进入训练，`Device: cuda, Num Parameters: 224061696`，loss 开始下降，无 NaN/报错。

- 根据最新用户优先级（先看 `I02/I03/I07` 三类 ID 的完整 scaling 现象），已调整运行队列：
  - 暂停 `vg_b_I06`，释放 GPU6 给当前主分析任务；`I06-diffgrm t5-base` 可后续从已有 checkpoint 继续 resume。
  - 新增 tmux=`sw_I03l`（GPU4）：顺序评测 `I03-seater / M03-t5l` 的 `4999 -> 9999 -> 14999 -> 18999`
  - 新增 tmux=`sw_I02_sparse`（GPU6）：顺序评测 `I02-semrq / {M01-t5s,M02-t5b,M03-t5l}` 的 `9999 -> 14999 -> 19999`
  - 原有 tmux=`sw_I07s`（GPU0）继续顺序评测 `I07-rkmeans / M01-t5s` 的 `9999 -> 14999 -> 19999`

- 截至当前已拿到的新结果：
  - `I03-seater / t5-large / step 4999`：
    - `R@5=0.0199; R@10=0.0331; N@5=0.0138; N@10=0.0183`
    - `unique_items=2218`
    - 与已有 `I03` 结果对比：`t5-small N@10=0.0112 (unique=2726)`，`t5-base N@10=0.0093 (unique=6017)`，`t5-large N@10=0.0183 (unique=2218)`。
    - 结论：`I03` 在 VG 上**并非单调不 scale**；至少在 `step 4999`，`t5-large` 明显优于 `t5-small/t5-base`，但 diversity 低于 small/base，提示其 gain 可能伴随更强的输出集中化。
  - `I02-semrq / t5-small / step 9999`：
    - `R@5=0.0175; R@10=0.0284; N@5=0.0123; N@10=0.0160`
    - `unique_items=2445`
  - `I02-semrq / t5-small / step 14999`：
    - `R@5=0.0110; R@10=0.0189; N@5=0.0073; N@10=0.0100`
    - `unique_items=5635`
    - 对照 `step 4999: N@10=0.0192, unique_items=114`，可见 `I02` 的后期行为不是“继续极端 collapse”，而是**diversity 持续上升、排序质量持续下降**；即从“极端集中但分数较高”转向“更分散但更差”。

- 当前待继续观察的队列：
  - `sw_I03l` 仍会继续跑 `9999/14999/18999`，用来判断 `I03-large` 是否也在 `4999` 达峰。
  - `sw_I02_sparse` 仍会继续补 `I02 t5-small@19999` 以及 `t5-base/t5-large @9999/14999/19999`。
  - `sw_I07s` 仍在跑 `I07 t5-small@9999`；完成后将自动续到 `14999/19999`。

### 2026-04-15

- 继续推进 VG 上其余三类 ID（`I04-semid / I05-letter / I06-diffgrm`）的 scaling 实验：
  - 远端空闲 GPU：`4/5/6`
  - 数据盘余量：`388G`，足够先开三路 `t5-base`
  - 当前目录状态核对：
    - `I04-semid`：仅有 `M01-t5s/checkpoint_4999.pt`
    - `I05-letter`：仅有 `M01-t5s/checkpoint_4999.pt`
    - `I06-diffgrm`：有 `M01-t5s/checkpoint_4999.pt`；`M02-t5b/decoder` 目录存在但此前无 checkpoint

- 已启动三路 `t5-base` 训练（统一使用 `configs/scaling_amz23vg_t5base.gin`）：
  - tmux=`vg_b_I04`（GPU4）：
    - 命令：`bash /tmp/vg_launch.sh I04-semid M02-t5b 4 /data/xqp_data/RecSys26/third_party/RQ_VAE_Recommender/configs/scaling_amz23vg_t5base.gin`
    - 日志：`/tmp/vg_b_I04.log`
    - 当前状态：已进入训练，约 `loss 10.19 -> 9.96`，step `~27/20000`
  - tmux=`vg_b_I05`（GPU5）：
    - 命令：`bash /tmp/vg_launch.sh I05-letter M02-t5b 5 /data/xqp_data/RecSys26/third_party/RQ_VAE_Recommender/configs/scaling_amz23vg_t5base.gin`
    - 日志：`/tmp/vg_b_I05.log`
    - 当前状态：已进入训练，约 `loss 8.42 -> 7.86`，step `~46/20000`
  - tmux=`vg_b_I06`（GPU6）：
    - 命令：`bash /tmp/vg_launch.sh I06-diffgrm M02-t5b 6 /data/xqp_data/RecSys26/third_party/RQ_VAE_Recommender/configs/scaling_amz23vg_t5base.gin`
    - 日志：`/tmp/vg_b_I06.log`
    - 当前状态：已进入训练，约 `loss 8.27 -> 7.75`，step `~45/20000`

- 同期补充到的已完成结果：
  - `I07-rkmeans / t5-small / step 9999`：`N@10=0.0116`
    - 对照 `step 4999: N@10=0.0196`，再次支持 `I07` 在 VG 上的 early-peak no-scaling 结论
    - 当前 `sw_I07s` 已自动切到 `step 14999` 导出中

- 根据最新磁盘管理策略，已执行两项清理：
  - 停止 `sw_I07s`：
    - 原因：`I07-rkmeans / t5-small` 已从 `4999: N@10=0.0196` 下降到 `9999: N@10=0.0116`，按当前策略不再继续 `14999/19999` sweep
    - 已删除半成品目录：`/tmp/sweep_vg_I07-rkmeans_M01-t5s/14999`
  - 删除已完整记录实验的冗余 checkpoint，仅保留每个 completed run 的最佳 checkpoint：
    - 保留：
      - `I02-semrq/M01-t5s/checkpoint_4999.pt`
      - `I02-semrq/M02-t5b/checkpoint_4999.pt`
      - `I02-semrq/M03-t5l/checkpoint_14999.pt`
      - `I03-seater/M03-t5l/checkpoint_4999.pt`
      - `I07-rkmeans/M02-t5b/checkpoint_4999.pt`
      - `I07-rkmeans/M03-t5l/checkpoint_4999.pt`
    - 保留未清理（因实验未完全结束或未来仍可能复查）：
      - `I03-seater/M01-t5s/*`
      - `I03-seater/M02-t5b/*`
      - `I07-rkmeans/M01-t5s/*`
    - 删除结果：共删 `66` 个 checkpoint，估算释放 `520.12 GB`
    - `df -h /data`：空闲空间从约 `388G` 回升到 `908G`

- 启动 `I04/I05/I06` 的首个 `t5-base@4999` 评测（与训练并行）：
  - tmux=`ev_b_I04`（GPU4）：
    - `bash /tmp/vg_eval.sh I04-semid M02-t5b 4999 4 t5-base`
    - 当前已进入 `Export preds`，约 `2/860`
  - tmux=`ev_b_I05`（GPU5）：
    - `bash /tmp/vg_eval.sh I05-letter M02-t5b 4999 5 t5-base`
    - 当前已进入 `Export preds`，约 `12/860`
  - tmux=`ev_b_I06`（GPU6）：
    - `bash /tmp/vg_eval.sh I06-diffgrm M02-t5b 4999 6 t5-base`
    - 当前已进入 `Export preds`，约 `7/860`
  - 观察：`I04` 的 vocab 最大（`ckpt_vocab=58083`，`sem_id_dim=9`），导出速度显著慢于 `I05/I06`

- `I04/I05/I06` 的 `t5-base` 训练已全部完成到 `20000/20000`：
  - `I04-semid / t5-base`：最后观察到 `step=20000`，`loss≈0.14`
  - `I05-letter / t5-base`：最后观察到 `step=20000`，`loss≈0.1258`
  - `I06-diffgrm / t5-base`：最后观察到 `step=20000`，`loss≈0.2472`

- `I04/I05/I06` 的 `t5-base@4999` 评测现已全部完成：
  - `I04-semid / t5-base / 4999`：
    - `R@5=0.0114; R@10=0.0205; N@5=0.0079; N@10=0.0109`
  - `I05-letter / t5-base / 4999`：
    - `R@5=0.0112; R@10=0.0194; N@5=0.0077; N@10=0.0105`
  - `I06-diffgrm / t5-base / 4999`：
    - `R@5=0.0205; R@10=0.0309; N@5=0.0147; N@10=0.0183`
  - 当前观察：
    - `I04/I05` 相比各自 `small@4999`（`0.0239 / 0.0324`）大幅下降
    - `I06` 也从 `small@4999=0.0279` 下降到 `base@4999=0.0183`，但仍显著高于 `I04/I05 base`

- 启动下一阶段研究准备：**码本利用率 vs decoder performance**
  - 协议更新：
    - 已将 `ScalingLaw_Protocol.md` 的诊断指标重构为两层：
      - tokenizer-side utilization / collision
      - decoder-side output bias / downstream performance
    - 新增独立章节，明确：
      - `LETTER` 对齐 `first-level code usage + frequency distribution`
      - `OneRec` 需区分 paper claim / public impl / local balanced reproduction
      - 研究问题必须回答 “tokenizer utilization 是否真的传导到 decoder 性能”
  - 日志/元数据工具：
    - 新增：`/Users/chenyufei/Documents/RecSys26/scripts/tokenizer_metrics_utils.py`
    - 新增：`/Users/chenyufei/Documents/RecSys26/scripts/export_tokenizer_metrics.py`
    - 新增：`/Users/chenyufei/Documents/RecSys26/scripts/prediction_bias_utils.py`
    - 新增：`/Users/chenyufei/Documents/RecSys26/scripts/export_prediction_bias.py`
    - 作用：统一输出
      - `tokenizer_meta.json`
      - `tokenizer_level_counts.json`
      - `tokenizer_full_code_counts.json`
      - `tokenizer_metrics_summary.json`
      - `prediction_bias_summary.json`
  - 已接入自动落盘的 tokenizer 脚本：
    - `/Users/chenyufei/Documents/RecSys26/OneRec/balanced_rqkmeans.py`
    - `/Users/chenyufei/Documents/RecSys26/scripts/scaling/residual_kmeans.py`
  - 说明：
    - `LETTER` 当前远端 build 脚本未直接 patch；后续可在生成 `index.json` 后用 `export_tokenizer_metrics.py` 进行同口径补录，无需重写 tokenizer 训练链路
  - 本地校验：
    - 已对新增/修改文件运行 `py_compile`，通过
    - 已对 tokenizer / decoder 两个工具做最小 smoke test，JSON 落盘格式正常

- 为“码本利用率 vs decoder performance”新增独立实验入口（不复用 `build_letter_indices_setrec.sh`）：
  - 新增脚本：`/Users/chenyufei/Documents/RecSys26/scripts/run_letter_codebook_experiment.py`
  - 设计目标：
    - 本地一条命令驱动远端 `LETTER tokenizer` 训练
    - 自动选 `best_collision_model.pth`
    - 自动生成 `index.json`
    - 自动导出 `tokenizer_meta.json / tokenizer_level_counts.json / tokenizer_full_code_counts.json / tokenizer_metrics_summary.json`
    - 可选导出 `prediction_bias_summary.json` 并收集 decoder metrics
  - 运行方式：本地 orchestrate 远端，不依赖旧的远端 `build_letter_indices_setrec.sh`
  - 本地校验：
    - `py_compile` 通过
    - `--help` 通过
    - `--dry-run --domain amazon23_vg --tag smoke` 输出的远端路径/命令符合当前目录规范

- 研究线口径调整（按最新决策冻结）：
  - `LETTER line`：继续使用 `LETTER` 官方 tokenizer 训练/生成链路，研究 diversity/collision-control 机制对 tokenizer 与 decoder 的影响
  - `GRID-RQ line`：统一改用 `snap-research/GRID` 的 `RQ-VAE` 与 `RQ-Kmeans` 作为 RQ family 的主实验实现
  - 当前 `I07-rkmeans` 等非 GRID 变体不再作为该研究线的主实验对象，仅保留为历史参考

- 继续推进 VG 上 `I04/I05/I06` 的 scaling：
  - 已于 `2026-04-16` 启动三路 `t5-large` 训练：
    - tmux=`vg_l_I04`（GPU4）
    - tmux=`vg_l_I05`（GPU5）
    - tmux=`vg_l_I06`（GPU6）
  - 当前状态：均已通过环境检查并进入 fresh start 初始化阶段，等待真正进入 loss 迭代后继续监控

- 完成 `semantic ID length scaling` 的两个 inference-only 诊断实验（2026-04-24）：
  - 新增脚本：
    - 本地：`/Users/chenyufei/Documents/RecSys26/scripts/length_inference_diagnostics.py`
    - 远端：`/data/xqp_data/RecSys26/scripts/length_inference_diagnostics.py`
  - 输入目录：
    - `/data/xqp_data/RecSys26/logs/code_length_study/amazon23_vg/20260421_clean_suffix_mainalign_t5small`
  - 输出目录：
    - 远端：`/data/xqp_data/RecSys26/logs/code_length_study/amazon23_vg/20260421_clean_suffix_mainalign_t5small/inference_diagnostics_20260424`
    - 本地：`/Users/chenyufei/Documents/RecSys26/tmp/inference_diagnostics_20260424`
  - 正式记录文档：
    - `/Users/chenyufei/Documents/RecSys26/ID_Length_Inference_Diagnostics_20260424.md`
  - 诊断 1：Token utilization / codebook collapse
    - `RQ-VAE` 出现明确 collapse：last-layer usage 从 `L2=43.0%` 降到 `L4=7.3%`、`L6=1.9%`、`L8=0.8%`、`L12=0.4%`；last-layer normalized entropy 也接近 `0`
    - `RQ-Kmeans` 没有 dead-token collapse（usage=100%），但后层 normalized entropy 从 `L2≈0.955` 降到 `L16≈0.844`，Gini 升到 `0.683`
    - `OPQ` 没有 collapse，usage=100%，normalized entropy 高且随长度整体更稳定
  - 诊断 2：Layer-wise information gain
    - 当前执行的是 `post-hoc prefix/mask projection`，不重训、不重新 beam search
    - `RQ-Kmeans` 大多在 `k=3~6` 后饱和，长码后续 token 几乎不再增加区分能力
    - `OPQ L6` 呈现最自然的逐层增益，`R@10` 从 `k1=0.0139` 逐步增至 `k6=0.0487`；`L>=8` 往往在 `k=5/6` 已饱和
  - 当前解释：
    - `RQ-VAE` 的长度退化主要来自 codebook collapse
    - `RQ-Kmeans` 是早期饱和，长码带来生成负担但边际信息弱
    - `OPQ L6` 是 tokenizer 不 collapse 且 decoder 可用性较好的折中点

- MicroLens-1M 上 `SEATER/LETTER` large-drop 诊断进入 native/official 阶段（2026-04-27）：
  - 背景：
    - `I03-SEATER` 和 `I05-LETTER` 的 decoder-friendly adapter sanity run 均未恢复正常 warm-only 指标，不能直接用于 small/base/large scaling 结论
    - 后续必须先验证官方/native tokenizer 或 native model 是否能在 MicroLens-1M 上产生非 collapse 结果
  - `SEATER`：
    - 原始 native paper profile 已跑通数据加载和 epoch 0 训练，但在 validation beam search 触发 `RuntimeError: CUDA error: invalid configuration argument`
    - 根因定位为 validation/export 的 `test_batch_size=2048` 与 `topK=50` 组合导致 `B*beam=102400`，进入 PyTorch SDPA attention 时 kernel 配置非法
    - 已新增 `paper_evalsafe` profile：训练 batch/epoch 保持 paper profile，validation/export `test_batch_size` 降到 `64`
    - 偏离记录：该改动只影响 validation/export batch size，不改变训练目标、模型结构、tree 数据或 topK；目的为避开 CUDA kernel 限制
  - `LETTER`：
    - `recsys26_letter` 通过显式 `PYTHONPATH` 复用 `recsys26_seater` 的 `k_means_constrained` 与 `ortools`
    - 已新增 `scripts/scaling/launch_microlens1m_letter_constrained_rebuild.sh`
    - 当前先用现有 checkpoint 做 constrained index rebuild，并输出 collision 统计；若 collision 仍维持高位，再决定是否完整重训 tokenizer
    - 结果更新：现有 checkpoint 的 constrained rebuild 已完成，但 `unique_codes=33227/88866`、`collision_rate=0.626100`、`max_collision_group=1170`，与旧结果一致；说明问题不是单纯 export 时缺 `k_means_constrained`，而是该 checkpoint 本身已高度碰撞
  - 详细进度和路径同步在：
    - `/Users/chenyufei/Documents/RecSys26/MicroLens1M_VGStyle_Scaling_20260425.md`

- MicroLens-1M 上 `LETTER` 官方 tokenizer 修复继续推进（2026-04-27）：
  - 已确认旧 checkpoint 的训练参数为 `alpha=0.1, beta=0.0001, epochs=500, eval_step=50`，其 `best_collision_rate=0.6819143429433079`，因此旧 index 的 `collision_rate=0.626100` 不是单纯导出问题。
  - 使用 `k_means_constrained` 后，官方 `RQ-VAE/models/vq.py` 的 `size_max=size_min*4` 在 `88866` 个 item 上约束不可行，容量为 `256*200=51200`，小于样本数。
  - 已在远端官方 LETTER copy 中把 constrained kmeans 的 `size_max` 改为 `max(original_bound, ceil(n_samples/n_clusters))`，同时保留 balanced-kmeans 约束语义。
  - 已重启官方量级修复训练：GPU1，`alpha=0.1, beta=0.1, epochs=500, batch_size=1024, eval_step=25`。
  - 日志：
    - launcher: `/data/xqp_data/RecSys26/logs/letter_tokenizer_setrec/microlens_1m/launchers/letter_constrained_full_gpu1_20260426_183707.nohup`
    - train: `/data/xqp_data/RecSys26/logs/letter_tokenizer_setrec/microlens_1m/train_20260426_183717.log`
  - 后续补充：
    - `KMeansConstrained` 的 VQ 初始化在 `88866` item 上过慢：`n_init=10` 约 14 分钟未过初始化；`n_init=1` 约 8 分钟仍未过初始化。
    - 已添加 `RECSYS26_LETTER_INIT_BACKEND=sklearn`，用于只把 VQ 初始化切回官方代码原有的 sklearn fallback 路径；训练目标仍保留 `beta` diversity loss。
    - `beta=0.1 + sklearn init` 已进入训练，前两次 collision eval 为 epoch 9 `0.936252`、epoch 19 `0.908019`，仍明显偏高。
    - 已并行启动隔离输出的 `beta=1.0 + sklearn init` 诊断 run：GPU3，launcher `/data/xqp_data/RecSys26/logs/letter_tokenizer_setrec/microlens_1m/launchers/letter_constrained_full_gpu3_20260426_191143.nohup`，train log root `/data/xqp_data/RecSys26/logs/letter_tokenizer_setrec/microlens_1m_beta1_sklearn_init/`。
    - 进一步诊断后，`beta=1.0` 在 epoch 9 的 collision 为 `0.964936`，已停止；`beta=0.1 + sklearn init + constrained trainer` 到 epoch 69 仍为 `0.806608`，且该 run 使用默认输出路径，已停止以避免覆盖 index；`beta=0.1 + sklearn init + sklearn trainer/export` 在 epoch 9 为 `0.939414`，也已停止。
    - 已把 `RECSYS26_LETTER_CLUSTER_BACKEND=sklearn` 同步到 `third_party/LETTER/RQ-VAE/trainer.py` 与 `scripts/build_letter_indices_setrec.sh`，保证训练评估和最终导出 backend 一致。
    - 当前唯一保留的 clean recovery run 是 `beta=1e-4 + sklearn init + sklearn trainer/export`，GPU3，launcher `/data/xqp_data/RecSys26/logs/letter_tokenizer_setrec/microlens_1m/launchers/letter_constrained_full_gpu3_20260426_193558.nohup`，train log `/data/xqp_data/RecSys26/logs/letter_tokenizer_setrec/microlens_1m_beta1e4_sklearn_all/train_20260426_193610.log`，checkpoint root `/data/xqp_data/RecSys26/datasets/letter_tokenizer_setrec/microlens_1m_beta1e4_sklearn_all`，index out `/data/xqp_data/RecSys26/tmp/letter_beta1e4_sklearn_all_index/SETRec_microlens_1m.index.json`。
    - 判断标准：epoch 49 collision 需要对齐或优于旧 run 的 `0.712421`；如果仍显著偏高，则 MicroLens-1M 上 LETTER 的官方 tokenizer 本身无法稳定产生低碰撞索引，不能继续做 decoder scaling 结论。
    - 已通过第一判断点：当前 clean run 的 epoch 49 collision 为 `0.707323`，略好于旧 run 同 epoch 的 `0.712421`。因此继续跑满 500 epoch，等待最终 checkpoint 与 isolated index 导出后再决定是否进入 decoder scaling。
    - 该 clean run 已完成 500 epoch 并导出 isolated index。最终训练侧 collision eval 为 epoch 499 `0.681577`，略好于旧 best `0.6819143429433079`。
    - 但是最终 isolated index 仍高度碰撞：`items=88866`, `unique_codes=33229`, `collision_rate=0.6260774649472239`, `max_collision_group=2157`, `per_position_usage=[124,256,256,255]`，路径 `/data/xqp_data/RecSys26/tmp/letter_beta1e4_sklearn_all_index/SETRec_microlens_1m.index.json`。
    - 结论更新：`beta=1e-4 + sklearn-all` 可以恢复旧 tokenizer 训练曲线，但没有真正修好 MicroLens-1M 的 LETTER index；下一步要定位 final index generation / collision repair 为什么只能得到约 `33k` unique codes。
    - 后续定位发现，MicroLens-1M 的源 T5 embedding 文件 `/data/xqp_data/RecSys26/third_party/SETRec/data/microlens_1m/microlens_1m.emb-t5-tdcb.npy` 本身只有 `35055/88866` 个 exact unique rows，重复率 `0.605529673891027`，最大 exact duplicate group 为 `1170`。最大 LETTER full-code group `2157` 个 item 在源 embedding 上四舍五入到 `1e-6` 后只有 `5` 个 unique rows，样本 pairwise L2 约 `1e-7`。
    - 因此当前 LETTER 的高 collision 主要受源 item embedding 重复限制；纯 embedding tokenizer 不加 item-level tie-breaker/dedup suffix 无法把这些 item 区分开。若要得到 decoder-usable 的 MicroLens-1M LETTER ID，需要显式 tie-breaker/dedup suffix，或换用重复率更低的 item representation。
    - `SEATER` native `paper_evalsafe` 已完成：
      - run root: `/data/xqp_data/RecSys26/logs/native_microlens1m/seater/20260426_181441__paper_evalsafe__gpu0`
      - checkpoint: `/data/xqp_data/RecSys26/logs/native_microlens1m/seater/20260426_181441__paper_evalsafe__gpu0/workspace/SEATER_SETRec_microlens_1m_paper_evalsafe__4_26_18/ckpt/best.pth`
      - warm-only full: `R@5=0.0064`, `N@5=0.0043`, `R@10=0.0118`, `N@10=0.0062`
      - warm-only loo: `R@5=0.0067`, `N@5=0.0040`, `R@10=0.0124`, `N@10=0.0058`
      - prediction diversity: `515434` exported users, `9673` unique top-1 items, `26535` unique top-10 items
      - 结论：SEATER 不是 LETTER 那种源 embedding exact-duplicate 导致 item 不可区分的问题；native 输出有一定多样性，但推荐指标本身过低。当前不应继续做 SEATER small/base/large decoder scaling，除非先做 SEATER-specific retuning 或把它作为 MicroLens-1M 上 native failure case。
    - 进一步完成 MicroLens-1M pipeline 根因排查：
      - `microlens_50k.emb-t5-tdcb.npy`：`14023/14079` exact unique rows，duplicate rate `0.003977555224092599`，最大重复组 `7`
      - `microlens_1m.emb-t5-tdcb.npy`：`35055/88866` exact unique rows，duplicate rate `0.605529673891027`，最大重复组 `1170`
      - `combine_tdcb_maps.npy` 本身不是主因：1M 现有 prompt `empty=0/88866`，`unique_texts=87036/88866`；用实际 builder parser 检查 `item_map_reverse` 到 `MicroLens-1M_title.csv` 覆盖为 `88866/88866`
      - 真实 root cause：MicroLens-1M raw title 是中文，而当前 pipeline 仍用英文 `sentence-transformers/sentence-t5-base`；不同中文标题被 tokenizer 映射成同一模式，例如 `['▁The','▁item','▁title','▁is','▁','<unk>','▁#','<unk>','▁#','<unk>','▁#','<unk>','.','</s>']`，从而生成大规模 identical embeddings
      - 因此当前 MicroLens-1M 上所有依赖 `microlens_1m.emb-t5-tdcb.npy` 的 text-embedding SID scaling 结果都只能作为 invalid diagnostic，不可用于正文结论；修复路线是先换中文/多语 sentence encoder 重建 item embeddings，再重建相关 tokenizer 和 decoder scaling
      - SEATER 路线需要单独处理：它用 SASRec item embedding，不受该 T5 中文 `<unk>` collapse 直接影响；但 1M 上 SASRec/SEATER native 本身指标偏低，需要 backbone/profile retuning 或单独标记为 native failure case
    - MicroLens-100K 数据准备完成（2026-04-27）：
      - 官方入口：`https://recsys.westlake.edu.cn/`，下载目录 `MicroLens-100k-Dataset`
      - 远端 raw path：`/data/xqp_data/RecSys26/datasets/microlens_100k/raw`
      - 已下载非媒体核心文件：`MicroLens-100k_pairs.csv`, `MicroLens-100k_pairs.tsv`, `MicroLens-100k_title_en.csv`, `tags_to_summary.csv`, `readme.txt`, `MicroLens-100k_likes_and_views.txt`, `MicroLens-100k_comment_en.txt`，总量约 `62M`
      - 远端 DNS 对 `recsys.westlake.edu.cn` 会错误解析到 `198.18.0.95`；实际下载时通过 `curl --resolve recsys.westlake.edu.cn:443:42.247.30.189` 指定公网 IP
      - SETRec split 已按 50K 同口径生成：`k_core=5`, `split_ratio=0.17`, `val_multiplier=1.8`, `min_train_len=2`, `item_shuffle_seed=2023`
      - 100K split manifest：`n_users=90171`, `n_items=17228`, `split_time1=1661735593066`, `split_time2=1660883466915`
      - `build_setrec_combine_tdcb_maps.py` 已加入 `microlens_100k` 白名单，并新增 `--microlens_title_fallback`；100K 有 `56` 个空英文标题，已用 `tags_to_summary.csv` 回填，最终 `missing_titles=0`, `fallback_used=56`
      - `combine_tdcb_maps.npy`：`17228` prompts，`empty=0`, `unique_texts=16971`, duplicate text rate `0.01491757603900623`
      - `microlens_100k.emb-t5-tdcb.npy` 已生成：shape `(17228,768)`, exact unique rows `16983/17228`, duplicate rate `0.014221035523566239`, 最大重复组 `12`
      - 判断：100K 使用英文标题，T5 embedding 重复率与 50K 同量级，不存在 1M 的中文 `<unk>` collapse；更适合作为 50K/100K scaling 对比数据集
    - MicroLens-100K 20k decoder gate 启动（2026-04-27）：
      - 新增远端脚本：`scripts/scaling/launch_microlens100k_20k_cell.sh` 与 `scripts/scaling/monitor_microlens100k_20k_queue.sh`
      - 评测脚本 `scripts/unified_eval.py` 已加入 `microlens_100k` 数据集选项，并沿用 warm-only 主评测口径
      - decoder raw adapter 已直接调用 `prepare_rqvae_recommender_raw` 生成，跳过通用 adapter 脚本里无关的 SEATER/SASRec 依赖
      - `I04-semid` 资产已生成：`cached_ids=(17228,4)`, `per_pos_sizes=[64,16015,8,60]`, `codebook_size=16015`
      - `I07-rkmeans` 资产已生成：balanced `3x256` + dedup suffix，`cached_ids=(17228,4)`, unique codes `16564/17228`, collision 由 suffix 修复为 `0`
      - GPU 状态：GPU0/1 被其他用户占用；GPU2 也被其他进程占用；GPU5 上有其他用户低利用率进程，占用约 `22.9GB`，已共享启动一个 small 训练
      - 当前运行：`I04-semid/M01-t5s`，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_20k/I04-semid/M01-t5s/20260427_132638__gpu3`
      - 已手动补启：`I07-rkmeans/M01-t5s`，GPU5，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_20k/I07-rkmeans/M01-t5s/20260427_135037__gpu5`
      - 队列监控：PID `4151624`，log `/data/xqp_data/RecSys26/logs/scaling_microlens100k_20k/launchers/monitor_queue_20260427_132625.nohup`；已写入 `queue_I07_M01_20k.started`，监控脚本于 `13:50:47` 退出，避免重复启动
      - 纠正记录（2026-04-27 15:00 后）：
        - 用户原意是 MicroLens-100K 上对比 `I05-LETTER` 与 `I07-RQKmeans`，不再测 `I03-SEATER`
        - `I04-semid/M01-t5s` 属于误启动的 sanity run，不进入 100K scaling 对比结论；其 warm-only full 为 `R@5=0.0081`, `N@5=0.0052`, `R@10=0.0146`, `N@10=0.0074`
        - `I07-rkmeans/M01-t5s` 训练已完成到 `checkpoint_19999.pt`；第一次导出被人工停止，已在 GPU3 重新补导出与 warm-only 评测
        - `I07-rkmeans/M01-t5s` 补导出完成，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_20k/I07-rkmeans/M01-t5s/20260427_135037__gpu5`；warm-only full: `R@5=0.0092`, `N@5=0.0053`, `R@10=0.0167`, `N@10=0.0079`; warm-only loo: `R@5=0.0088`, `N@5=0.0048`, `R@10=0.0165`, `N@10=0.0074`
        - `I05-letter` 资产已生成：`SETRec_microlens_100k.index.json` 中 `17228/17228` unique codes，collision rate `0`; `tiger_compat` 为 `cached_ids=(17228,4)`, `per_pos_sizes=[1,1,68,256]`
        - 当前运行：`I05-letter/M01-t5s`，GPU5，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_20k/I05-letter/M01-t5s/20260427_150750__gpu5`；截至 `2026-04-27 15:35` 已保存 `checkpoint_4999.pt`，训练约 `9130/20000`
        - 进一步排查发现，100K 这批 20k sanity launcher 误用了 50K 旧 `base+UID` 口径：`scaling_ml50k_t5small_base.gin` + `train.num_user_tokens=2000`。50K clean `v5` 实际使用 `scaling_ml50k_t5small_grid_aligned_fast.gin` + `train.num_user_tokens=1`（TIGER-noUID/grid-aligned）。因此上述 100K `I04/I07` 结果不是与 50K clean `v5` 同口径结果，不应解释为单纯数据规模翻倍导致掉点。
        - 已停止错误口径的 `I05-letter/M01-t5s/20260427_150750__gpu5` 训练；未删除产物，保留为 invalid diagnostic。
        - 已修正 `scripts/scaling/launch_microlens100k_20k_cell.sh` 的 `M01-t5s` 分支：改用 `scaling_ml50k_t5small_grid_aligned_fast.gin`，并将 `train.num_user_tokens=1`。
        - 已重新启动同口径 20k sanity：
          - `I07-rkmeans/M01-t5s`：GPU3，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_20k/I07-rkmeans/M01-t5s/20260427_154435__gpu3`
          - `I05-letter/M01-t5s`：GPU5，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_20k/I05-letter/M01-t5s/20260427_154435__gpu5`
          - 两者日志均确认 `include scaling_ml50k_t5small_grid_aligned_fast.gin`、`train.num_user_tokens=1`，已进入训练。
        - 修正口径 20k sanity 已完成（2026-04-27 18:40 复查，无相关训练/导出进程仍在运行）：
          - `I07-rkmeans/M01-t5s/20260427_154435__gpu3`：warm-only full `R@5=0.0103`, `N@5=0.0068`, `R@10=0.0160`, `N@10=0.0087`; warm-only loo `R@5=0.0109`, `N@5=0.0072`, `R@10=0.0157`, `N@10=0.0088`
          - `I05-letter/M01-t5s/20260427_154435__gpu5`：warm-only full `R@5=0.0025`, `N@5=0.0015`, `R@10=0.0043`, `N@10=0.0021`; warm-only loo `R@5=0.0025`, `N@5=0.0015`, `R@10=0.0037`, `N@10=0.0019`
          - 解释：`I07-rkmeans` 修正后已回到接近 MicroLens-50K v5 的量级；`I05-letter` 在同口径下仍明显异常，下一步应优先比较 50K/100K 的 `I05` 码本结构、prefix 利用率、dedup suffix 与导出候选覆盖，而不是继续用旧 base+UID 结果解释数据规模效应。
        - `I05-letter` 异常根因已确认（2026-04-28）：
          - `third_party/LETTER/data/SETRec_microlens_100k/SETRec_microlens_100k.index.json` 不是由官方 LETTER tokenizer 训练链路生成；`logs/letter_tokenizer_setrec/` 和 `datasets/letter_tokenizer_setrec/` 下没有 `microlens_100k` 的 tokenizer 训练日志或 checkpoint。
          - 实际来源是 `scripts/prepare_setrec_adapters.py::prepare_letter()` 的占位逻辑，代码注释明确为 `Placeholder index file: unique 4-token codes in base-256`。该逻辑按 item id 顺序写 `<a_0>, <b_0>, <c_*>, <d_*>`，因此 100K 产物退化为 `per_pos_sizes=[1,1,68,256]`，前两层恒定，不是 learned LETTER code。
          - 之前尝试生成 100K `SASRec_item_embed.pkl` 的任务失败，错误为缺少 `third_party/SEATER/config/SETRec_microlens_100k/SASREC.yaml`；因此当时并不具备官方 LETTER tokenizer 所需的 `cf_emb`。
          - 已修复远端配置：复制 50K 的 `SASREC.yaml` 到 `third_party/SEATER/config/SETRec_microlens_100k/SASREC.yaml`，并将 `scripts/build_letter_indices_setrec.sh` 的 domain 白名单加入 `microlens_100k`。
          - 已启动真实 LETTER 修复 pipeline：PID `356719`，脚本 `logs/letter_tokenizer_setrec/microlens_100k/repair/run_repair_pipeline.sh`，主日志 `logs/letter_tokenizer_setrec/microlens_100k/repair/latest.nohup`。流程为先在 GPU2 训练 `SETRec_microlens_100k` 的 SASRec 并导出 `SASRec_item_embed.pkl`，再在 GPU3 运行 `build_letter_indices_setrec.sh microlens_100k`，最后重建 `logs/scaling_microlens100k_20k/I05-letter/{manifest,tiger_compat}`。
          - 当前旧的 `I05-letter/M01-t5s/20260427_154435__gpu5` 结果应标记为 invalid diagnostic，不能作为 MicroLens-100K LETTER scaling 结果；需要等待真实 LETTER index 和 compat 重建完成后重新训练 decoder。
        - `I05-letter` 官方修复 pipeline 已完成（2026-04-28 复查）：
          - SASRec item embedding 已生成：`third_party/SETRec/data/microlens_100k/SASRec_item_embed.pkl`，shape=`(17228,64)`。
          - 真实 LETTER tokenizer/index 已生成：`third_party/LETTER/data/SETRec_microlens_100k/SETRec_microlens_100k.index.json`；生成日志报告 `All indices number=17228`, `Max number of conflicts=18`, `Collision Rate=0.04208265614116555`。
          - raw LETTER compat 的 `cached_ids.npy` 为 `(17228,4)`，仅 `16503` 个唯一 full-code，存在 `725` 个碰撞；因此不能直接作为 decoder clean result。
          - 已将 raw compat 备份为 `logs/scaling_microlens100k_20k/I05-letter/tiger_compat.raw_letter_bak_*`，并用 `scripts/scaling/build_letter_split_dedup_compat.py` 追加 1 位 `base=256` dedup suffix：新 `cached_ids.npy` 为 `(17228,5)`，唯一 full-code=`17228/17228`，碰撞率=`0`，`per_pos_sizes=[38,107,163,256,256]`。
          - 已启动修复后 `I05-letter/M01-t5s` 20k sanity run：GPU3，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_20k/I05-letter/M01-t5s/20260427_213329__gpu3`；日志确认 `sem_ids_dim=5`, `duplicate_policy=unique`, `train.num_user_tokens=1`，正在训练。
        - 修复后 `I05-letter/M01-t5s` 20k sanity 已完成：checkpoint=`checkpoint_19999.pt`，warm-only full `R@5=0.0087`, `N@5=0.0057`, `R@10=0.0156`, `N@10=0.0081`; warm-only loo `R@5=0.0096`, `N@5=0.0060`, `R@10=0.0149`, `N@10=0.0077`。该结果已明显脱离旧 placeholder 的异常低值，但 full `R@5` 仍低于同口径 `I07-rkmeans/M01-t5s` 的 `0.0103`。
        - MicroLens-100K train / MicroLens-50K fixed eval 协议已落地（2026-04-28）：
          - 目的：隔离“训练数据从 50K 扩到 100K”对 small/base/large scaling 的影响，避免直接 50K vs 100K 评测时同时改变 candidate pool、test users/items 与 split cutoff。
          - 新增脚本：`scripts/scaling/build_microlens100k_train50k_eval.py`, `scripts/scaling/eval_fixed_setrec.py`, `scripts/scaling/launch_microlens100k_train50k_eval_20k_cell.sh`。
          - `scripts/scaling/export_with_manifest.py` 新增 `--testing_dict_path`，用于 warm-only export 时显式指定固定 eval user set。
          - 新 domain：`third_party/SETRec/data/microlens_100k_train_50k_eval`。训练侧使用 `MicroLens-100k_pairs.csv`，但按 MicroLens-50K cutoff 重切：`split_time1=1661747100482`, `split_time2=1660903181455`，从而避免 100K cutoff 对 50K fixed eval 的泄漏。
          - item map 严格复用 `microlens_100k`，因此可直接复用 100K 的 tokenizer/cached IDs；user map 在此基础上追加 116 个 50K eval 用户，保证 fixed eval 不丢用户。
          - 构建校验：`n_users_output_map=90287`, `n_items_source_map=17228`, train interactions=`487407`, val interactions=`78314`, test interactions=`93033`, train warm items=`13976`。
          - fixed 50K eval 校验：`eval50_users=42666`, `eval50_gt_items=44430`, `eval50_warm_items=11346`, `eval50_warm_gt_users=6482`, `eval50_warm_gt_items=8031`, `missing_eval50_users_in_100=0`, `missing_eval50_items_in_100=0`, `eval50_warm_items_in_full_train_warm_rate=1.0`。
          - RQ-VAE/TIGER decoder raw adapter 已生成：`datasets/rqvae_recommender/setrec/raw/microlens_100k_train_50k_eval`。
          - 已启动第一条 sanity run：`I07-rkmeans/M01-t5s`，GPU3，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M01-t5s/20260428_125537__gpu3`，launcher log `/data/xqp_data/RecSys26/logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M01-t5s_launch_20260428_2055.log`。
          - 该 run 使用 100K `I07-rkmeans/tiger_compat`，训练配置为 `scaling_ml50k_t5small_grid_aligned_fast.gin`, `train.num_user_tokens=1`, `train.iterations=20000`；导出配置为 fixed eval users + `topk_items=100`，评测脚本会把 predictions 过滤到 `eval50_warm_item.npy` 后计算 warm-only full/loo。
          - `I07-rkmeans/M01-t5s` sanity run 已完成训练与重导出。第一次导出因 `top100 + batch_size=192` 跑到 GPU0 且显存不足 OOM；已用同一 `checkpoint_19999.pt` 在 GPU3 重导出，保持 `beam_size=100`, `num_return_sequences=100`, `topk_items=100`，仅将 export batch size 降为 `16`。
          - fixed-50K warm candidate 评测结果：full `R@5=0.0177`, `N@5=0.0117`, `R@10=0.0262`, `N@10=0.0146`; loo `R@5=0.0178`, `N@5=0.0113`, `R@10=0.0254`, `N@10=0.0137`。
          - 评测校验：full evaluated users=`6482`, GT items=`8031`, predictions users=`6482`, `missing_users=0`, `short_users_after_candidate_filter=0`; loo evaluated users=`4337`, `missing_users=0`, `short_users_after_candidate_filter=0`。
          - 基于上述结果，已启动同协议的 `I07-rkmeans` base/large scaling：
            - `M02-t5b`：GPU3，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M02-t5b/20260428_225316__gpu3`，launcher log `/data/xqp_data/RecSys26/logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/launchers/M02-t5b_gpu3_20260429.log`。
            - `M03-t5l`：GPU6，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M03-t5l/20260428_225316__gpu6`，launcher log `/data/xqp_data/RecSys26/logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/launchers/M03-t5l_gpu6_20260429.log`。
          - 已将 `scripts/scaling/launch_microlens100k_train50k_eval_20k_cell.sh` 的 fixed-eval export batch size 调整为 `small=16`, `base=8`, `large=2`，保留 `top100` 导出和 fixed 50K warm candidate 过滤，降低导出 OOM 风险。
          - `I07-rkmeans/M02-t5b` 已完成训练、导出和 fixed-50K warm candidate 评测。run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M02-t5b/20260428_225316__gpu3`。
          - `I07-rkmeans/M02-t5b` 结果：full `R@5=0.0215`, `N@5=0.0153`, `R@10=0.0352`, `N@10=0.0199`; loo `R@5=0.0198`, `N@5=0.0130`, `R@10=0.0332`, `N@10=0.0173`。评测校验均为 `missing_users=0`, `short_users_after_candidate_filter=0`。
          - `I07-rkmeans/M03-t5l` 仍在训练：截至 2026-04-29 约 `9350/20000`，已保存 `checkpoint_4999.pt`，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_train50k_eval_20k/I07-rkmeans/M03-t5l/20260428_225316__gpu6`。
          - 已启动 `I05-letter/M01-t5s` 同协议 fixed-50K eval scaling：GPU1，run root `/data/xqp_data/RecSys26/logs/scaling_microlens100k_train50k_eval_20k/I05-letter/M01-t5s/20260429_064517__gpu1`，launcher log `/data/xqp_data/RecSys26/logs/scaling_microlens100k_train50k_eval_20k/I05-letter/launchers/M01-t5s_gpu1_20260429.log`。
          - `I05-letter/M01-t5s` 启动日志已确认使用修复版 100K LETTER compat：`cached_ids=(17228,5)`, `per_pos_sizes=[38,107,163,256,256]`, `sem_id_dim=5`, `duplicate_policy=unique`；训练 domain 为 `microlens_100k_train_50k_eval`，不是 direct 100K eval。
          - 2026-04-29 进度更新：`I07-rkmeans/M03-t5l` 仍在 GPU6 训练，截至复查约 `14444/20000`，已保存 `checkpoint_4999.pt` 与 `checkpoint_9999.pt`，暂无导出和 metrics。
          - 2026-04-29 进度更新：`I05-letter/M01-t5s` 已训练完成到 `checkpoint_19999.pt`，第一次 `top100` 导出完成，但 strict fixed-eval 失败：`missing_users=0`, `short_users=337`，即 337 个用户在 warm candidate 过滤后不足 top-10 预测；因此该 top100 结果不作为有效 metrics。
          - 2026-04-29 已启动 `I05-letter/M01-t5s` 同 run 的补导出：`beam_size=300`, `num_return_sequences=300`, `topk_items=300`, `batch_size=4`, 输出 `export/pred_topk_top300_eval50.jsonl`，日志 `export_top300.log`；待该补导出完成后再生成 `metrics_{full,loo}_warm_only_eval50_candidate_top300.json`。
