# Phase 3：两遍式文献综述与研究思路

Phase 3 的目标是让系统完成一组可审计的文献综述工作：先逐篇抽取统一字段，再跨论文比较，最后提出带证据边界和可证伪实验的研究假设。它不把 50 篇论文一次性塞进 Prompt，也不把模型生成的“新颖”当成事实。

## 1. 当前 MVP 工作流

```text
研究主题 + 论文 aliases
  -> 每个 alias 独立 BM25 检索（exact-alias 过滤）
  -> 每篇论文独立切分 E1.S1 等可信 span
  -> Pass 1：抽取 method / bitstream / hardware / evaluation / limitations
  -> 应用层绑定 fact_id、alias、chunk、PDF 页码和原文
  -> Pass 2：只输入已验证 fact records
  -> 生成五维比较表、冲突项和最多三个研究假设
  -> 应用层校验 fact_id，并重新绑定所有 citation
```

每篇论文至少一次 LLM 调用，综合阶段再调用一次；结构化输出修复最多增加一次调用。Pass 2 看不到原始 PDF 文本，只能引用 Pass 1 产生的 `F1`、`F2` 等事实 ID，这是防止综合阶段越过证据边界的关键设计。

## 2. 运行命令

先同步代码并确认测试：

```bash
git pull origin main
make test
```

运行一个低离散序列的小型综述：

```bash
make literature-review \
  TOPIC="低离散位流生成器的精度、硬件成本与并行化" \
  ALIASES="Sobol PSobol P2LSG FSM-LDSG" \
  REVIEW_OUTPUT=artifacts/phase3_low_discrepancy.json
```

也可以直接调用 CLI：

```bash
.venv/bin/sc-rag-review \
  "随机计算中相关性操纵与精度权衡" \
  --aliases CorrManip Isolation ZCE Contingency \
  --output artifacts/phase3_correlation.json
```

输出中的核心字段：

- `papers`：每篇论文的五个统一维度；每个事实带 `fact_id` 和完整 citation；
- `comparison`：每个维度的论文单元格和显式 `conflicts`；空单元格表示语料或检索证据没有覆盖；
- `ideas`：`observed_gap → inferred_limitation → hypothesis → proposed_experiment → falsifier → uncertainty`；
- `missing_aliases`、`missing_dimensions`：明确披露没有成功进入综述的论文或字段。

查看某个比较维度：

```bash
jq '.comparison[] | select(.dimension == "hardware")' artifacts/phase3_low_discrepancy.json
jq '.ideas[] | {gap: .observed_gap, hypothesis, experiment: .proposed_experiment, falsifier}' artifacts/phase3_low_discrepancy.json
```

## 3. 对应代码和理解顺序

| 工作 | 代码 | 需要理解的原因 |
|---|---|---|
| 论文级候选检索 | `BM25Index.search_alias` | exact alias 过滤解决 `Sobol`/`PSobol` 名称碰撞 |
| Pass 1 Prompt | `EXTRACTION_SYSTEM_PROMPT`、`extraction_prompt` | 每篇论文有独立证据边界，缺失字段保留为空 |
| 事实编号 | `LiteratureReviewService.review` | 应用层生成 `F1` 等 ID，模型不能伪造元数据 |
| 事实引用绑定 | `citation_for_span`、`merge_citations` | citation 来自检索 span，不来自模型自由填写的页码 |
| Pass 2 Prompt | `SYNTHESIS_SYSTEM_PROMPT`、`synthesis_prompt` | 综合只能使用验证后的事实记录，避免二次幻觉 |
| 事实 ID 校验 | `_resolve_facts` | 未知 fact、跨论文错绑 fact 都 fail-closed |
| 研究假设 | `_bind_ideas` | 事实、推断、假设和可证伪条件分开返回 |
| HTTP 接口 | `POST /literature-review` | 后续 UI 可以直接展示论文卡片、比较表和假设 |

## 4. 人工验收 rubric

先抽查一组 4 篇论文的输出，再进入更大规模综述：

1. **Traceability**：每个比较单元格的 fact ID 能回到 alias、页码和原文 span；
2. **Completeness**：没有证据的维度显示为空，而不是模型补写；
3. **Conflict preservation**：不同技术节点、数据集或位流长度的数字没有被合并；
4. **Feasibility**：每个研究假设有可执行实验和明确 falsifier；
5. **Uncertainty**：`novelty_status` 保持 `unverified`，除非人工完成新颖性检索。

Phase 3 的第一个验收不是“生成了漂亮的综述”，而是能随机挑选 10 个比较单元格和全部研究假设，逐项复核来源和推理边界。

## 5. 面试要能解释的问题

- 为什么文献综述要采用 map-then-synthesize，而不是一个超长 Prompt？
- Pass 1 的结构化事实和 Pass 2 的比较表分别解决什么错误？
- 如何保留不同论文实验条件导致的数字冲突？
- 为什么研究假设不能直接标成“创新点”？
- 如果一篇论文没有覆盖 hardware 字段，系统应如何呈现？
