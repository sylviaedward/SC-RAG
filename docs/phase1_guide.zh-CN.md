# Phase 1 操作与代码导读：Hybrid Retrieval

Phase 1 的目标不是简单增加一个向量数据库，而是用同一批题目回答四个可测量的问题：BM25 找到了什么、Dense 找到了什么、两路融合是否互补、Reranker 是否把真正支持问题的证据排到前面。

## 1. Codespaces 运行顺序

先更新代码并安装 CPU 推理依赖：

```bash
git pull origin main
make setup-hybrid
```

`setup-hybrid` 会先从 PyTorch 官方 CPU 源安装 CPU wheel，再安装 Sentence Transformers。我们不下载 CUDA 运行库，因为当前 Codespaces 没有使用 GPU。

建立 Dense 索引：

```bash
make dense-index
```

首次运行会从 Hugging Face 下载 `intfloat/multilingual-e5-small`。随后程序将 2,216 个 chunk 编码成归一化向量，并生成：

```text
artifacts/dense/embeddings.npy
artifacts/dense/metadata.json
```

`metadata.json` 保存模型名、chunk 文件 SHA256、chunk ID 顺序和矩阵形状。只要 chunking 发生变化，旧 Dense 索引就会拒绝加载，防止向量与文本错位。

用同一个问题依次运行四种检索：

```bash
make search QUERY="随机计算中的相关性如何影响乘法器精度？"
make dense-search QUERY="随机计算中的相关性如何影响乘法器精度？"
make hybrid-search-rrf QUERY="随机计算中的相关性如何影响乘法器精度？"
make hybrid-search QUERY="随机计算中的相关性如何影响乘法器精度？"
```

四条命令分别代表：

1. BM25；
2. multilingual-E5 Dense Retrieval；
3. BM25 + Dense + RRF；
4. RRF 候选 + Cross-Encoder Reranker + 论文多样性控制。

第一次执行 `make hybrid-search` 会下载 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`。观察结果中的：

- `bm25_rank`：该 chunk 在 BM25 路径中的名次；
- `dense_rank`：该 chunk 在 Dense 路径中的名次；
- `rrf_score`：只由两路名次计算的融合分数；
- `reranker_score`：Cross-Encoder 对问题与 chunk 联合编码后的相关性分数；
- `rank`：当前阶段的最终名次。

字段按阶段填写，并非每条命令都会填满：纯 BM25 只有 `bm25_rank`，纯 Dense 只有 `dense_rank`，RRF 才有 `rrf_score`，完整 Hybrid 才有 `reranker_score`。在 Hybrid 中，某个 chunk 只被一路召回时，另一路的 rank 仍会是 `null`，这正好用于观察两路召回的互补性。

先运行无需 Cross-Encoder 的三阶段评测：

```bash
make evaluate-phase1-core
```

它在相同 50 道可回答题上依次评价 BM25、Dense 和 Hybrid RRF，结果写入 `artifacts/phase1_core_eval.json`。随后单独运行资源消耗更高的精排评测：

```bash
make evaluate-phase1-reranker
```

结果写入 `artifacts/phase1_reranker_eval.json`。两段都会每 5 题显示进度，并输出：

```text
Hit-any@10
Hit-all@10
MRR
mean latency
p95 latency
mean returned hits
full-K rate
```

`mean_returned_hits` 和 `full_k_rate` 用于检查多样性过滤后是否仍返回完整 Top-10。`make evaluate-phase1` 可顺序运行两段，但在 Codespaces 中建议先分别执行，便于观察 CPU 和内存。如果精排被系统以 `Error 143` 终止，前三阶段结果仍然安全保留；运行 `free -h` 检查内存，再根据最后一条进度定位。资源较小的 Codespace 可以先执行 `SC_RAG_TORCH_THREADS=1 make evaluate-phase1-reranker RERANK_K=10`，确认链路后再恢复 `RERANK_K=30`。`RERANK_K=10` 只用于链路验证，不能作为正式消融结论。不要只看总分：至少找出 BM25 成功但 Dense 失败、Dense 成功但 BM25 失败、Reranker 改善、Reranker 伤害这四类样本。

## 2. 工作流和代码对应关系

```text
问题
  ├─→ 中文术语扩展 → BM25 Top-40 ──────────┐
  └─→ E5 query embedding → cosine Top-40 ──┤
                                             → RRF 去重与融合
                                             → 取 Top-30
                                             → Cross-Encoder 精排
                                             → 每篇论文最多 2 个 chunk
                                             → Top-K 证据
```

| 步骤 | 解决的问题 | 对应代码 |
|---|---|---|
| Passage embedding | 将中英文语义映射到同一向量空间 | `src/sc_rag/dense.py:passage_text`、`SentenceTransformerEmbedder` |
| 持久化索引 | 避免每次查询重新编码 2,216 个 chunk | `build_dense_artifacts` |
| 版本校验 | 防止新的 chunks 配上旧向量 | `chunks_sha256`、`DenseIndex.from_artifacts` |
| Dense Top-K | 用归一化向量点积得到 cosine similarity | `DenseIndex.search` |
| RRF | 融合量纲不同的 BM25 分数和 cosine 分数 | `reciprocal_rank_fusion` |
| Reranker | 对少量候选联合编码，获得更精确的 query-passage 判断 | `CrossEncoderReranker.score` |
| 论文多样性 | 防止同一篇论文的相邻 chunks 占满 Top-K | `apply_paper_diversity` |
| 完整编排 | 控制 candidate_k、rerank_k、RRF 和输出数量 | `HybridRetriever.search` |
| 消融评测 | 在完全相同题目上比较各阶段收益与延迟 | `scripts/evaluate_phase1.py` |

## 3. 为什么不直接相加两种分数

BM25 分数没有固定上界，cosine similarity 通常落在较小区间，Cross-Encoder 又有自己的 logit 标度。直接写 `bm25_score + dense_score` 会让某一路因为数值范围更大而支配结果，还会随模型或语料变化。

RRF 只使用名次：

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

一个 chunk 同时被两路排在前面时会获得更高分；只被其中一路召回时仍有机会进入候选。`k=60` 用于降低第一名与后续名次之间的过度差距。

## 4. Dense Retrieval 与 Reranker 的区别

Dense Retrieval 分别编码问题和全部 chunks，文档向量可以离线保存，因此查询时只需一次 query embedding 和矩阵乘法，适合高召回。

Cross-Encoder 将“问题 + 候选 chunk”一起送入模型，能看到两者细粒度交互，通常排序更准，但每个候选都要单独推理。因此这里只重排融合后的 Top-30，而不是扫描 2,216 个 chunks。

## 5. 本阶段需要亲手做的实验

### 实验 A：互补性

选择三个问题，记录 BM25 和 Dense 的 Top-5。解释哪些结果来自精确关键词，哪些来自语义匹配。至少找出一个只有其中一路召回正确论文的例子。

### 实验 B：候选规模

在 `HybridRetriever.__init__` 中将 `candidate_k` 从 40 改成 10，运行 `make evaluate-phase1-core`；再恢复成 40。观察召回、延迟和 Reranker 输入数量的关系。

### 实验 C：论文多样性

将 `max_per_paper` 从 2 改成 1，再评测。解释为什么跨论文问题可能受益，而需要同篇论文多个证据块的问题可能受损。

### 实验 D：Reranker 失败

对比 `hybrid_rrf` 与 `hybrid_rrf_reranked` 的逐题结果，找到至少一个名次改善和一个名次下降的样本。检查是模型语义判断、文本截断、公式解析还是 gold alias 粒度造成的。

## 6. 可选的 Embedding API

默认本地模型便于复现。如果 Codespaces 已安全配置 `EMBED_BASE_URL`、`EMBED_API_KEY`、`EMBED_MODEL_NAME`，也可以运行：

```bash
make dense-index EMBEDDING_PROVIDER=api
make dense-search EMBEDDING_PROVIDER=api QUERY="你的问题"
make evaluate-phase1 EMBEDDING_PROVIDER=api
```

不要在终端命令、`.env`、代码或 Git 中写入 key，应使用 Codespaces Secrets。API 和本地模型生成的索引不可混用，metadata 中的模型名必须匹配。

## 7. 进入 Phase 2 的门槛

完成以下内容后再进入有证据回答：

1. 能解释 bi-encoder、cosine similarity、RRF、cross-encoder 的区别；
2. 能从 CLI 输出说明一个 chunk 来自哪条召回路径、为什么改变名次；
3. 有四阶段消融表和至少四类失败样本；
4. 能解释 `candidate_k=40`、`rerank_k=30`、`max_per_paper=2` 的成本和风险；
5. 能证明 Dense 索引与 chunks 版本绑定，不会静默错位；
6. 能回答为什么更强的 Reranker 仍可能降低某些检索指标。

