# SC-RAG 功能工作流与代码导读

这份文档回答三个问题：当前功能怎样运行、数据经过了什么处理、每一步为什么存在。先完整跑通 Phase 0，再按代码导读逐步理解；遇到不懂的实现可以直接带着文件名、函数名和现象来讨论。

## 一、在 GitHub Codespaces 里先跑什么

打开仓库的 Codespace 后，在终端执行：

```bash
make setup
make test
```

这一步不需要论文 PDF。它验证 Python 环境、包安装、50 篇论文元数据、58 道评测题的数据契约和一个小型检索样例。

接着创建 `imports/`，把 `SC_Liter1.zip` 到 `SC_Liter4.zip` 拖到 Codespaces 左侧文件区的该目录，然后执行：

```bash
mkdir -p imports
make import-corpus ARCHIVES="imports/SC_Liter*.zip"
make manifest
make ingest
```

预期输出依次是：

- `unique_pdfs=50`：51 个文件中跳过 1 个重复文件；
- `wrote 50 unique paper records`：文件名、哈希、页数和 DOI 与清单一致；
- `wrote 2216 page-aware chunks`：建立本地 `artifacts/chunks.jsonl` 检索数据。

现在运行第一条真实检索：

```bash
make search QUERY="随机计算中的相关性如何影响乘法器精度？"
```

重点观察返回结果中的 `alias`、`title`、`page`、`text`、`score` 和 `rank`。然后运行全量基线：

```bash
make evaluate
```

最后启动 API：

```bash
make api
```

在 Codespaces 弹出的端口提示中打开 8000 端口，访问 `/docs` 使用 Swagger；或者另开一个终端执行：

```bash
curl http://127.0.0.1:8000/health
curl --get --data-urlencode 'q=低差异序列如何提高随机计算精度？' \
  --data 'limit=5' http://127.0.0.1:8000/search
```

建议第一次只按上面的顺序执行，不要直接运行某个 `.py` 文件。`Makefile` 是操作入口，具体 Python 文件是随后阅读和调试的对象。评测问题和答案 JSONL 已在 GitHub 中，不需要为了运行基线再次上传原始评测 PDF；只有在研究题库抽取逻辑时才需要把该 PDF 放入 `data/eval/source/`。

## 二、Phase 0：数据治理与 BM25 检索

### 工作流 A：论文进入知识库

```text
4 个 ZIP
  → 安全解压并按 SHA256 去重
  → 校验 50 个规范文件名
  → 读取 PDF 页数、文本量和 DOI
  → 按 PDF 页提取文本
  → 页面内分块并保留重叠
  → 写入 chunks.jsonl
```

对应代码：

| 步骤 | 为什么 | 代码入口 |
|---|---|---|
| ZIP 导入 | 避免手工解压错误和路径穿越，并自动跳过重复论文 | `scripts/import_corpus_archives.py:import_archives` |
| 语料清单 | 用 SHA256、页数和规范别名固定这一次实验使用的数据版本 | `scripts/build_corpus_manifest.py:build_manifest` |
| PDF 提取 | 保留 PDF 页码，让后续证据可以回到原文 | `src/sc_rag/ingest.py:iter_chunks` |
| 页面内分块 | 控制检索粒度；重叠用于减少段落边界造成的语义断裂 | `src/sc_rag/ingest.py:split_page` |
| 稳定 ID | 同一论文、页码、块序号得到稳定身份，便于引用和回归比较 | `src/sc_rag/ingest.py:iter_chunks` |

你应该亲手做的实验：把 `target_chars` 从 1400 改成 700，重新运行 `make ingest && make evaluate`，比较 chunk 数和指标，再改回去。这个实验会直接说明 chunk size 为什么不是随便设置的。

### 工作流 B：用户问题变成检索结果

```text
中文问题
  → SC 中英文术语扩展
  → 英文单词/数字/中文二元词切分
  → 对每个 chunk 计算 BM25
  → 按分数排序
  → 返回带论文和页码的 Top-K 证据
```

对应代码：

| 步骤 | 为什么 | 代码入口 |
|---|---|---|
| 术语扩展 | 问题主要是中文，论文主要是英文，纯关键词无法直接匹配 | `src/sc_rag/text.py:TERM_EXPANSIONS`、`expand_query` |
| Tokenize | 把文本转换成 BM25 可以统计的词项 | `src/sc_rag/text.py:tokenize` |
| 建立统计量 | 预先计算 TF、DF、IDF、文档长度，避免每次重复准备 | `src/sc_rag/search.py:BM25Index.__init__` |
| BM25 排序 | 兼顾词项稀有度、词频饱和和文档长度 | `src/sc_rag/search.py:BM25Index.search` |
| API 暴露 | 让 UI 或其他 Agent 通过稳定接口使用检索能力 | `src/sc_rag/api.py:search` |

调试建议：在 `expand_query`、`tokenize` 和 `BM25Index.search` 设置断点，用同一个问题观察原问题、扩展问题、tokens 和最终分数。

### 工作流 C：题库如何评估检索

```text
58 道题的题目文件 + 独立答案文件
  → 排除 8 道不可回答题
  → 对 50 道可回答题逐题检索 Top-10
  → 将结果中的论文 alias 与 gold_aliases 比较
  → 汇总 Hit-any、Hit-all 和 MRR
```

对应代码：

| 步骤 | 为什么 | 代码入口 |
|---|---|---|
| 题目/答案分离 | 防止 gold answer 被索引或进入生成上下文造成数据泄漏 | `scripts/extract_eval_benchmark.py` |
| Alias 对齐 | 当前先评价是否找对论文，不让答案生成质量干扰检索评测 | `scripts/evaluate_retrieval.py:evaluate` |
| Hit-any | 判断是否至少找到一篇支撑论文 | `scripts/evaluate_retrieval.py:evaluate` |
| Hit-all | 判断跨论文问题所需论文是否全部找齐 | `scripts/evaluate_retrieval.py:evaluate` |
| MRR | 评价第一篇正确论文出现得是否足够靠前 | `scripts/evaluate_retrieval.py:evaluate` |

当前 Top-10 基线是 Hit-any 0.86、Hit-all 0.58、MRR 0.7569。下一阶段首先要解决 Hit-all 较低的问题。

## 三、Phase 1：混合检索

计划工作流：

```text
问题
  ├─→ BM25 稀疏检索 ─┐
  └─→ Dense 向量检索 ─┤
                       → RRF 融合并去重
                       → Reranker 精排
                       → 论文多样性约束
                       → Top-K 证据
```

为什么需要它：BM25 擅长精确术语、论文名和公式符号，Dense Retrieval 擅长语义相似和中英文表达差异；RRF 避免直接混合量纲不同的分数；Reranker 用更强模型判断“问题—证据”相关性。对应模块会在进入 Phase 1 时新增，并与当前 `BM25Index` 保持相同接口，方便做消融实验。

进入下一阶段的门槛：你能独立跑通 Phase 0，能解释 ingestion、BM25 和三个指标，并完成一次 chunk size 实验。不是按日期推进。

## 四、Phase 2：有证据的回答与拒答

计划工作流：

```text
问题
  → 意图/问题类型识别
  → 混合检索与精排
  → 证据充足度判断
  ├─ 不足：明确拒答并说明语料未覆盖
  └─ 充足：构造受约束上下文
           → LLM 生成结构化回答
           → 逐条验证引用是否来自提供的 chunk
           → 返回回答、引用、置信度和 trace
```

该阶段会新增模型适配器、Prompt、结构化输出、引用校验器和 `/ask`。需要重点理解：检索置信度不等于答案正确；LLM 说“参考了某论文”不等于引用真的支撑该陈述。

进入下一阶段的门槛：能解释一次回答从问题到引用的完整 trace；八道不可回答题有独立拒答结果；API key 不进入代码、Git 或日志。

## 五、Phase 3：文献综述与研究思路

计划工作流：

```text
研究主题
  → 检索候选论文并按条件过滤
  → 每篇论文独立抽取方法、数据、硬件和限制
  → 标准化为带引用的结构化记录
  → 跨论文比较，保留冲突和实验条件差异
  → 从“已知证据 → 限制 → 假设 → 可证伪实验”生成研究思路
```

这里使用两遍处理，而不是把 50 篇论文一次塞入一个 Prompt。第一遍保证每篇论文的证据边界，第二遍才综合。研究思路必须区分论文明确结论、系统推断和待验证假设，不能把“可能的新颖”写成“首次提出”。

进入下一阶段的门槛：比较表中每个关键单元格能回到论文页码；冲突数字不会被静默合并；至少一个研究思路包含可执行、可证伪的实验。

## 六、Phase 4：评测与可观测性

计划工作流：

```text
固定数据集与配置
  → 批量运行完整 RAG
  → 分别计算检索、答案、引用、拒答指标
  → 按问题类别生成回归报告
  → 保存 query rewrite、候选、精排、上下文、输出、延迟和成本 trace
  → 对失败样本归因
```

这一阶段把“Demo 看起来能用”变成“有证据证明哪些部分有效”。进入下一阶段前，应能用一个命令复现实验，并解释至少十个失败案例属于数据、解析、检索、重排、生成还是评测问题。

## 七、Phase 5：产品化与展示

计划工作流：

```text
Streamlit 研究工作台
  → 调用 FastAPI
  → 展示回答、页码证据、论文比较和运行 trace
  → Docker 固化依赖
  → 健康检查、缓存、错误处理和演示脚本
```

最终演示至少覆盖：一个基础知识问题、一个跨论文比较和一个正确拒答。到这里再整理架构图、指标表、README 截图、简历描述和两分钟项目介绍。

