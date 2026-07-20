# 外部资源与补充实验计划

本文档覆盖论文新增实验所需的外部资源、执行计划和发布前待处理问题。论文当前版本见
[`papers/RecSys_2026_paper_777.pdf`](../papers/RecSys_2026_paper_777.pdf)，其
SHA-256 为
`fc44412babcbf11db0496851e93ce4942773f6834ab0b0bb1e03933cd3986d5e`。

> 当前状态：源码布局和核心分析入口已通过语法检查，但本仓库尚不是“干净克隆后，
> 只下载数据集和 checkpoint 即可一键运行”的发布版本。具体阻塞项见第 7、8 节。

## 1. 范围与复用原则

当前论文中可直接复用的结果包括：

- RQ2：Video Games 上 `TIGER`、`RQ-Kmeans`、`LETTER-div`、
  `LETTER-no-div` 的结果。
- RQ4：Video Games 上六种 SID 的 `Jaccard@20` 和 `RBO@20`。
- RQ3：MicroLens-50K 上 `TIGER`、`RQ-Kmeans`、`LETTER-div` 的 T5-small
  推荐结果，以及 MicroLens-100K 上同协议的已有 scaling 结果；对应组合可作为
  RQ2 的下游性能列复用。

本文不要求重新下载或重跑上述结果。论文第 4.1 节将 “MicroLens” 默认定义为
MicroLens-50K，因此论文主表中的 “MicroLens” 指 `microlens_50k`；本轮补充实验
则将 RQ2 同时扩展到 `microlens_50k` 和 `microlens_100k`。

## 2. 需要下载的外部资源

先在仓库根目录初始化路径：

```bash
source scripts/env.sh
mkdir -p "${DATA_ROOT}/raw" "${MODEL_ROOT}"
```

### 2.1 数据集

| 数据集 | 本轮用途 | 必需的原始文件 | 放置位置 | 官方来源 |
| --- | --- | --- | --- | --- |
| MicroLens-50K | RQ2、RQ4 | `MicroLens-50k_pairs.csv`、`MicroLens-50k_titles.csv` | `${DATA_ROOT}/raw/microlens_50k/` | [MicroLens 下载站](https://recsys.westlake.edu.cn/)、[官方仓库](https://github.com/westlake-repl/MicroLens) |
| MicroLens-100K | RQ2、RQ4 | `MicroLens-100k_pairs.csv`、`MicroLens-100k_title_en.csv` | `${DATA_ROOT}/raw/microlens_100k/` | [MicroLens 下载站](https://recsys.westlake.edu.cn/)、[官方仓库](https://github.com/westlake-repl/MicroLens) |
| Yelp Open Dataset | RQ4 | 官方 `yelp_dataset.tar` 中的 `yelp_academic_dataset_review.json`、`yelp_academic_dataset_business.json` | 归档放在 `${DATA_ROOT}/raw/yelp/yelp_dataset.tar`；两个 JSON 解压到 `${DATA_ROOT}/raw/yelp/extracted/` | [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) |

MicroLens 可由仓库脚本直接下载；只下载本轮实际使用的交互和文本元数据：

```bash
bash scripts/run_data.sh download \
  --dataset microlens_50k \
  --output_root "${DATA_ROOT}/raw" \
  --files MicroLens-50k_pairs.csv MicroLens-50k_titles.csv

bash scripts/run_data.sh download \
  --dataset microlens_100k \
  --output_root "${DATA_ROOT}/raw" \
  --files MicroLens-100k_pairs.csv MicroLens-100k_title_en.csv
```

Yelp 的归档 URL 由官网页面动态提供。先从官网获得 `yelp_dataset.tar`，再执行：

```bash
bash scripts/run_data.sh download \
  --dataset yelp \
  --output_root "${DATA_ROOT}/raw" \
  --local_archive /absolute/path/to/yelp_dataset.tar \
  --extract
```

解压后必须确认两个 JSON 的最终路径与上表一致；若归档带额外顶层目录，应移动或
软链接到 `${DATA_ROOT}/raw/yelp/extracted/`，不要修改配置中的数据语义。

### 2.2 预训练模型

| 模型 | 本轮用途 | 本地目录 | 官方来源 |
| --- | --- | --- | --- |
| `sentence-transformers/sentence-t5-base` | 三个新数据集的统一 768 维文本表征、RQ4 reference space，以及内容型 SID 输入 | `${MODEL_ROOT}/sentence-t5-base/` | [Hugging Face 模型页](https://huggingface.co/sentence-transformers/sentence-t5-base) |
| `google-t5/t5-small` | 用于补齐两种 MicroLens 上缺失的 RQ2 受控推荐结果，尤其是 `LETTER-no-div` | `${MODEL_ROOT}/t5-small/` | [Hugging Face 模型页](https://huggingface.co/google-t5/t5-small) |

推荐显式下载到稳定目录，避免不同运行节点各自缓存一份：

```bash
hf download sentence-transformers/sentence-t5-base \
  --local-dir "${MODEL_ROOT}/sentence-t5-base"

hf download google-t5/t5-small \
  --local-dir "${MODEL_ROOT}/t5-small"
```

若最终只报告 RQ2 的 tokenizer 统计、不补齐缺失的受控下游推荐性能，`t5-small`
可以省略；完整回答 RQ2 时不建议省略这些结果。

### 2.3 明确不需要下载的资源

- 不需要重新下载 Video Games 数据或其已有模型、SID 和结果。
- 不需要 MicroLens 的视频、图像、音频、评论、likes/views 或预提取多模态特征。
- 不需要 Yelp photos、user、tip、check-in 数据；官方 JSON 归档中只使用 review 和
  business 两个文件。
- 不需要 `t5-base`、`t5-large`、LLaMA 或扩散模型权重。
- 不需要下载六种 SID 的第三方预训练 checkpoint。仓库中的 `baselines/` 已包含源代码；
  新数据集上的 tokenizer/checkpoint 应由统一划分和固定配置生成。

## 3. 下载资源与生成产物的目录边界

下面的文件是代码运行后生成的产物，不是外部下载资源，也不应提交到 GitHub：

```text
${DATA_ROOT}/
├── raw/
│   ├── microlens_50k/
│   ├── microlens_100k/
│   └── yelp/
├── setrec_data/
│   ├── microlens_50k/
│   ├── microlens_100k/
│   └── yelp/
└── seater_setrec/
    ├── microlens_50k/
    ├── microlens_100k/
    └── yelp/

${MODEL_ROOT}/
├── sentence-t5-base/
├── t5-small/
└── hf/

${TOKENIZER_ROOT}/
├── microlens_50k/{TIGER,SEATER,HowToIndex,LETTER,OPQ,RQ-Kmeans}/
├── microlens_100k/{TIGER,SEATER,HowToIndex,LETTER,OPQ,RQ-Kmeans}/
└── yelp/{TIGER,SEATER,HowToIndex,LETTER,OPQ,RQ-Kmeans}/

${RUN_ROOT}/supplementary/
├── rq2/{microlens_50k,microlens_100k}/
└── rq4/{amazon23_vg,microlens_50k,microlens_100k,yelp}/k_{10,20,50}/
```

多数据集实验必须使用 `${TOKENIZER_ROOT}/${dataset}/${method}`，不能沿用旧的
`${TOKENIZER_ROOT}/${method}` 单数据集布局，否则不同数据集的 `cached_ids.npy`
会互相覆盖。

每个 `${DATA_ROOT}/setrec_data/${dataset}/` 至少应生成：

- `training_dict.npy`、`validation_dict.npy`、`testing_dict.npy`
- `item_map.npy`、`item_map_reverse.npy`
- `warm_item.npy`、`cold_item.npy`
- `combine_tdcb_maps.npy`
- `${dataset}.emb-t5-tdcb.npy`

`LETTER` 和 `SEATER` 使用的 `SASRec_item_embed.pkl` 及对应 `best.pth` 是基于统一
交互划分训练得到的产物，不是需要下载的预训练模型。已有 RQ1 产物可以复用，但必须
先验证数据集、item map、训练配置和 checkpoint 完全一致。

## 4. 数据准备协议

所有新增实验保持论文中的统一设置：5-core 过滤、全局时间顺序划分、比例
`70:13:17`、`item_shuffle_seed=2023`、最大历史长度 50。配置来源为
`configs/data/setrec_splits.yaml`。

数据划分命令模板如下：

```bash
bash scripts/run_data.sh split \
  --format microlens \
  --input "${DATA_ROOT}/raw/microlens_50k/MicroLens-50k_pairs.csv" \
  --out_root "${DATA_ROOT}/setrec_data" \
  --dataset microlens_50k

bash scripts/run_data.sh split \
  --format microlens \
  --input "${DATA_ROOT}/raw/microlens_100k/MicroLens-100k_pairs.csv" \
  --out_root "${DATA_ROOT}/setrec_data" \
  --dataset microlens_100k

bash scripts/run_data.sh split \
  --format yelp \
  --input "${DATA_ROOT}/raw/yelp/extracted/yelp_academic_dataset_review.json" \
  --out_root "${DATA_ROOT}/setrec_data" \
  --dataset yelp
```

随后依次运行 `build-text` 和 `build-embeddings`。MicroLens-100K 与 Yelp 必须显式
传入各自的 metadata 路径；文本向量统一使用本地
`${MODEL_ROOT}/sentence-t5-base`。不要使用下载的多模态特征替代该 reference space。

预处理完成后，统计量应与论文表 1 完全一致：

| 数据集 | Users | Items | Interactions |
| --- | ---: | ---: | ---: |
| MicroLens-50K | 42,666 | 14,079 | 310,531 |
| MicroLens-100K | 90,171 | 17,228 | 658,056 |
| Yelp | 221,039 | 109,326 | 3,531,723 |

若任一统计量不一致，应停止后续实验并检查原始文件版本、字段和过滤流程，不能通过改
表格或截断数据来对齐。

## 5. RQ2：MicroLens-50K/100K codebook utilization

### 5.1 实验矩阵

数据集为 `microlens_50k` 和 `microlens_100k`，每个数据集分别比较四种 SID：

- `TIGER`：3 层 RQ-VAE，单层 codebook size 为 256。
- `RQ-Kmeans`：3 层 balanced residual K-means，单层 256。
- `LETTER-div`：保留 diversity regularization。
- `LETTER-no-div`：只关闭 diversity regularization；其他 tokenizer 设置保持一致。

MicroLens-50K 上 `TIGER`、`RQ-Kmeans`、`LETTER-div` 的 T5-small 推荐结果可从
论文 RQ3 复用。MicroLens-100K 的已有 scaling 结果也可逐项复用，但前提是 SID、
tokenizer checkpoint、T5-small 受控 backbone 和评测口径完全一致；其余缺失组合
需要新增 decoder 训练与评测。不得用 RQ1 的 method-native 分数替代 RQ2/RQ3 的
受控 TIGER-backbone 分数。

### 5.2 指标定义

所有统计在去掉 decoder 专用 de-dup suffix 后的原始三层 semantic codes 上计算。
令第 `l` 层 token `c` 的出现次数为 `n_l(c)`，总 item 数为 `N`，codebook 容量为
`C_l=256`：

- `used codes`：`|{c: n_l(c)>0}|`，逐层报告。
- `entropy`：`H_l=-sum_c p_l(c) log p_l(c)`；同时保存
  `H_l/log(C_l)`，主表优先报告 normalized entropy。
- `perplexity`：`exp(H_l)`；同时可报告 `perplexity/C_l` 作为 effective usage。
- `Gini`：在完整 256 维频数向量上计算，必须包含未使用 code 的零频数；值越低越均衡。
- `collision`：`(N - #unique raw semantic IDs)/N`。不能在追加唯一 suffix 后计算，
  否则 collision 会被人为变成 0。
- `prefix utilization@d`：
  `#unique prefixes of length d / min(N, product_{l<=d} C_l)`，报告 `d=1,2,3`。

现有 `experiments/rq2/tokenizer_metrics_utils.py` 已导出 entropy、perplexity、Gini、
collision 和 prefix 统计，但正式运行前必须用上述定义做一次单元核对，特别检查：

1. Gini 是否把 unused codes 的零频数纳入计算；
2. collision 是否排除了 de-dup suffix；
3. prefix utilization 的分母是否使用上述 capacity-adjusted 定义。

建议输出两张表，而不是把所有数字塞进一张表：

1. 每个 method × level 的 `used / normalized entropy / perplexity / Gini`；
2. 每个 method 的 `collision / prefix utilization@1,2,3 / R@10 / NDCG@10`。

机器可读结果统一放在
`${RUN_ROOT}/supplementary/rq2/${dataset}/${method}/`，至少保留
`tokenizer_meta.json`、`tokenizer_level_counts.json`、
`tokenizer_prefix_counts.json` 和 `tokenizer_metrics_summary.json`。

## 6. RQ4：跨数据集 local semantic preservation

### 6.1 实验矩阵

主表覆盖四个数据集：

- Video Games：复用已有 `K=20`，只补 `K=10,50`。
- MicroLens-50K、MicroLens-100K、Yelp：新增 `K=10,20,50`。

每个数据集比较六种 SID：`TIGER`、`SEATER`、`HowToIndex`、`LETTER`、`OPQ`、
`RQ-Kmeans`。三套新数据集优先复用已经通过 RQ1 生成且 item map 完全一致的 SID；
缺失项再按 `configs/tokenizer/main_table_tokenizers.yaml` 生成。

### 6.2 固定口径

- reference space：`sentence-t5-base` item-text embeddings，cosine distance。
- SID distance：`TIGER/SEATER/HowToIndex/LETTER/RQ-Kmeans` 使用 prefix distance，
  `OPQ` 使用 Hamming distance。
- de-dup suffix：按 `configs/experiment/rq4_geometry.yaml` 的
  `drop_last_code_column` 设置处理。
- 指标：`Jaccard@K` 和 `RBO@K`，`K in {10,20,50}`；RBO persistence 固定
  `p=0.9`。
- anchors：每个数据集固定 `anchor_samples=128`、`seed=20260423`，同一数据集的
  六种方法和三个 K 必须使用同一组 anchors。
- tie policy：沿用已提交 Video Games 结果的冻结实现，不在不同数据集或 K 之间切换
  conventional/tie-aware 口径。

论文当前文字称 RQ4 为 “across all items”，但冻结配置和现有结果实际使用 128 个
固定随机 anchors。为复用已有 Video Games 结果，本轮统一采用 128 anchors，并在论文
中改写为 sampled-anchor average。若决定真正改为 all-item evaluation，则必须先实现
分块 exact retrieval，并连同 Video Games 一起重跑，不能把两种口径放在同一张表中。

每个数据集和 K 单独运行并写入独立目录。例如：

```bash
bash scripts/run_rq4.sh \
  --reference_embeddings "${DATA_ROOT}/setrec_data/${dataset}/${dataset}.emb-t5-tdcb.npy" \
  --sid "TIGER=${TOKENIZER_ROOT}/${dataset}/TIGER/cached_ids.npy" \
  --sid "SEATER=${TOKENIZER_ROOT}/${dataset}/SEATER/cached_ids.npy" \
  --sid "HowToIndex=${TOKENIZER_ROOT}/${dataset}/HowToIndex/cached_ids.npy" \
  --sid "LETTER=${TOKENIZER_ROOT}/${dataset}/LETTER/cached_ids.npy" \
  --sid "OPQ=${TOKENIZER_ROOT}/${dataset}/OPQ/cached_ids.npy" \
  --sid "RQ-Kmeans=${TOKENIZER_ROOT}/${dataset}/RQ-Kmeans/cached_ids.npy" \
  --output_dir "${RUN_ROOT}/supplementary/rq4/${dataset}/k_${K}" \
  --neighbors_k "${K}" \
  --anchor_samples 128 \
  --rbo_persistence 0.9 \
  --paper_sid_metric native \
  --native_metric_override TIGER=prefix \
  --native_metric_override SEATER=prefix \
  --native_metric_override HowToIndex=prefix \
  --native_metric_override LETTER=prefix \
  --native_metric_override OPQ=hamming \
  --native_metric_override RQ-Kmeans=prefix \
  --drop_last_override TIGER=true \
  --drop_last_override SEATER=true \
  --drop_last_override HowToIndex=false \
  --drop_last_override LETTER=true \
  --drop_last_override OPQ=true \
  --drop_last_override RQ-Kmeans=true \
  --seed 20260423
```

`${dataset}` 和 `${K}` 分别替换为当前数据集名和 `10/20/50`；命令显式传入全部
distance 与 suffix 规则，不能依赖方法名的隐式默认值。

结果组织为：

1. 主表：四个数据集在 `K=20` 下的六种 SID × `Jaccard/RBO`；Video Games 行复用。
2. 敏感性结果：每个数据集分别绘制 `K=10,20,50` 的 Jaccard 和 RBO 曲线；同时保留
   完整 CSV，避免只在图中呈现。

## 7. 当前可运行性边界

截至本文档更新时，仓库状态如下：

| 层级 | 当前状态 | 结论 |
| --- | --- | --- |
| Canonical Python/Bash 源码 | `scripts/verify_source_layout.py --syntax` 通过 | 源码语法和目录布局可用 |
| `run_data.sh`、`run_rq1.sh`–`run_rq4.sh` | `--help` 均可启动 | 入口存在，但不等于端到端复现已闭环 |
| 严格资源检查 | 缺少 `datasets/`、`models/`、`envs/`、`logs/` | 当前服务器上的 source-only 目录不能直接开跑 |
| RQ2 分析入口 | 接受 `cached_ids.npy` 或 `index.json` | 不能直接读取普通训练 checkpoint |
| RQ4 分析入口 | 接受 reference embedding 和六份 `cached_ids.npy` | 不能只给六个方法 checkpoint 后直接分析 |
| 干净 GitHub 克隆 | 尚未验证 | 当前远程目录没有 `.git`，第三方恢复流程也未闭环 |

因此，“下载数据集和 checkpoint”并不是完整前置条件。实际还需要：

1. 创建正确的方法级运行环境；
2. 将原始数据转成统一 split 和 item map；
3. 生成 item text、`sentence-t5-base` embeddings 和 SASRec/CF embeddings；
4. 从 checkpoint 导出、或直接重新生成各数据集的 semantic IDs；
5. 将所有方法统一转成与 item map 同序的 `cached_ids.npy`；
6. 完成第 8 节的配置和指标修正后再运行补充实验。

如果已经拥有最终的 `${dataset}.emb-t5-tdcb.npy` 和全部 `cached_ids.npy`，RQ4
分析主体可以运行；RQ2 脚本虽然可以执行，但新指标在修正和单元测试通过前不能作为
论文结果使用。

## 8. 发布前必须处理的问题

### 8.1 GitHub 与第三方代码恢复

当前远程目录没有 `.git`，尚未执行 `git init`、commit 或 push。此外：

- `.gitignore` 排除了 `/third_party_clean/` 和 `/third_party`；
- `scripts/verify_source_layout.py` 又要求 `third_party_clean/` 中存在 11 个指定 commit
  的 Git worktree；
- 当前没有 bootstrap 脚本根据 `repos_manifest.tsv` 自动恢复这些 worktree。

发布前应保留 34 MB 的适配后源码 `baselines/`，继续忽略约 935 MB 的嵌套 Git
worktree；同时新增可重复执行的第三方 bootstrap 脚本，按 `repos_manifest.tsv` clone、
checkout 固定 commit，并应用 `patches/` 中的适配。否则干净克隆会在仓库自己的布局
检查中失败。

### 8.2 环境不是单一 requirements 可以解决

根目录 `requirements/environment.yml` 适合公共 pipeline，但不能覆盖所有原始
baseline。现有 baseline 对 PyTorch/Transformers 的要求存在明显冲突，例如
PyTorch `1.8.1`、`>=2.5.1` 和 `2.6.0` 同时存在。

发布前需要：

- 明确 `requirements/environment.yml` 只负责公共 data/RQ2/RQ4 pipeline；
- 为每种方法指定对应的 `patches/*_env.yml`；
- 提供 environment matrix 和创建/激活命令；
- 至少在目标 CUDA 驱动上完成一次逐环境 import smoke test。

### 8.3 补充实验配置仍是旧的单数据集配置

当前正式配置尚未落实本文档中的实验矩阵：

- `configs/experiment/rq2_codebook_usage.yaml` 仍固定 `dataset: amazon23_vg`，且使用
  `${TOKENIZER_ROOT}/${method}`，会造成多数据集输出覆盖；
- `configs/experiment/rq4_geometry.yaml` 仍固定 Video Games、`K=20` 和 Video Games
  的 case-study anchor `18089`；该 item id 对两个 MicroLens 数据集越界；
- `scripts/run_all.sh` 只把字符串参数转发给各入口，不会检查或执行
  download → split → embedding → tokenizer → analysis 的依赖关系。

发布前应增加 dataset-aware supplementary configs，将 RQ2 固定为
`{microlens_50k,microlens_100k} × 4 methods`，将 RQ4 固定为
`4 datasets × 6 methods × K{10,20,50}`，并统一使用
`${TOKENIZER_ROOT}/${dataset}/${method}`。

### 8.4 MicroLens-100K 的方法适配尚未完整

已核实的缺口包括：

- `baselines/ref01/config/SETRec_microlens_100k/` 不存在，因此 SEATER/SASRec 缺少
  MicroLens-100K 配置；
- `scripts/rq1_native/repro_seater.sh`、
  `scripts/rq1_native/repro_diffgrm_paperalign.sh` 和
  `scripts/rq1_native/repro_llm_id.sh` 的入口校验会拒绝 `microlens_100k`；
- canonical tokenizer pipeline 目前没有覆盖六种 RQ4 SID 的统一 dataset × method
  生成/导出入口。

发布前需要补齐 MicroLens-100K 配置和适配，确保六种方法均能产生 item 数量、顺序
和 semantic-level 定义一致的 `cached_ids.npy`。不得通过复制 MicroLens-50K
checkpoint 或截断数组来绕过数据集适配。

### 8.5 RQ2 新指标实现存在已复现的口径错误

`experiments/rq2/tokenizer_metrics_utils.py` 当前存在三项必须修正的问题：

1. `collision_rate` 基于包含 de-dup suffix 的完整 code 计算，会把原始 semantic-ID
   collision 人为变为 0；
2. Gini 只使用已出现 token 的频数，没有把 unused codes 的零频数纳入；
3. 当前 `prefix_uniqueness_rate` 的分母是 item 数，不完全等于第 5.2 节定义的
   capacity-adjusted prefix utilization。

合成数据测试中，两件具有相同三层 semantic code、仅 suffix 不同的 item，当前脚本
输出 `collision_rate=0.0`，正确的 raw semantic collision 应为 `0.5`；同一测试中
level-0 Gini 输出 `0.0`，纳入 255 个 unused codes 后应为 `0.996094`。

发布前必须修正实现，并增加至少以下单元测试：

- 全部 item 使用同一 code；
- codebook 完全均匀使用；
- raw semantic code 冲突、suffix 唯一；
- 不同 prefix depth 下的已知唯一前缀数；
- 有/无 suffix 两种输入产生相同的 semantic-level 指标。

### 8.6 需要真正的端到端发布验收

只有以下流程在新目录或新机器上通过后，才能称为“下载资源后可运行”：

1. 从 GitHub 干净克隆；
2. 根据 manifest 恢复第三方 audit worktrees；
3. 创建公共环境和实际用到的方法级环境；
4. 下载第 2 节列出的最小数据和模型；
5. 对一个小数据集完成 split、text、embedding、SID 生成和 RQ2/RQ4 smoke run；
6. 对 MicroLens-50K、MicroLens-100K、Yelp 完成完整输入验证；
7. 运行 RQ2 指标单元测试和 RQ4 item-order/shape 检查；
8. 执行 `scripts/verify_source_layout.py --syntax --strict-resources` 并通过。

## 9. 执行顺序与最终验收标准

1. 先完成第 8.1–8.5 节的阻塞修复，并在干净克隆中通过 source-only 检查。
2. 下载三套原始数据和两个模型，记录来源 URL、下载日期和 SHA-256。
3. 生成统一划分、item map、文本和 `sentence-t5-base` embeddings；核对表 1 统计量。
4. 对已有 SID 做 item-order hash 和 shape 核对；只生成缺失的 dataset × method 组合。
5. 完成 MicroLens-50K 和 MicroLens-100K 各四种 RQ2 tokenizer 统计；逐项复用
   同协议结果，只补缺失的受控推荐结果。
6. 完成 RQ4：三套新数据的三个 K，以及 Video Games 的 `K=10,50`。
7. 汇总机器可读结果，更新论文表格/图；不改写历史原始结果文件。

最终验收必须满足：

- 干净 GitHub 克隆能够恢复所需第三方代码和环境，不依赖服务器历史目录。
- 三个新数据集的统计量与论文表 1 一致。
- 每个 RQ4 输入满足
  `reference_embeddings.shape[0] == cached_ids.shape[0] == #items`，且 item 顺序 hash
  一致。
- 两种 MicroLens 上的 RQ2 四种方法均有完整三层指标，并按 raw semantic codes
  计算 collision。
- RQ2 合成单元测试覆盖 suffix、unused code 和 prefix utilization，并全部通过。
- RQ4 汇总包含 `4 datasets × 6 methods × 3 K = 72` 行，每行同时有 Jaccard 和 RBO。
- `datasets/`、`models/`、`logs/`、checkpoint 和 tokenizer 产物保持在 `.gitignore`
  范围内；GitHub 只同步源代码、配置、文档和论文 PDF。
