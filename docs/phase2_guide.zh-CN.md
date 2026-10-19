# Phase 2：证据约束回答操作与代码导读

Phase 2 的目标不是“让 LLM 看起来会回答”，而是让系统只交付能追溯到检索 chunk、证据 span 和 PDF 页码的 claim，并在证据不足、引用伪造或输出损坏时拒答。

## 1. 当前最小闭环

```text
问题 + 回答模式
  -> Hybrid RRF 检索候选池 Top-30
  -> 从问题正文解析已知的 [论文别名]
  -> 对候选池缺失的显式别名执行 exact-alias BM25 补召回
  -> alias-aware evidence planner 优先为每个显式别名保留至少一个 chunk
  -> 再按原检索顺序填充证据（最多 10 chunks / 14000 字符）
  -> 应用层把 chunk 切为 E1.S1 等可信证据 span
  -> 将论文内容标记为不可信数据
  -> OpenAI-compatible Chat Completions
  -> 严格 JSON 解析
  -> JSON/schema 失败或漏引已提供的显式目标时，执行最多一次定向修复
  -> span ID 白名单校验
  -> 应用层从 span ID 取回原文摘录
  -> 应用层绑定 alias、chunk ID、PDF 页码
  -> 完整回答、带缺失项的部分回答，或显式拒答
```

回答支持三种显式模式和一种自动判断模式：

| 模式 | 使用场景 | 生成约束 |
|---|---|---|
| `fundamentals` | SC 定义、公式、基础电路 | 保留公式、条件和取值范围 |
| `paper_lookup` | 指定论文或实现查询 | 只报告该论文明确给出的属性 |
| `comparison` | 跨论文比较 | 按共同维度比较，保留冲突和缺失 |
| `auto` | 交互式查询 | 根据“比较”等词和 `[论文别名]` 推断 |

## 2. 配置模型 API

代码按以下优先级读取配置：

1. `SC_RAG_LLM_BASE_URL`、`SC_RAG_LLM_API_KEY`、`SC_RAG_LLM_MODEL`
2. `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL_ID`
3. `DEEPSEEK_BASE_URL`、`DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL_ID`

不要把密钥写进仓库。优先使用 Codespaces Secrets。若 OpenAI 中转和 DeepSeek 两组变量同时存在，而本次希望明确使用 DeepSeek，可将 DeepSeek 的值映射到 `SC_RAG_LLM_*`。`BASE_URL` 应指向兼容 API 的 `/v1` 根路径；代码会追加 `/chat/completions`。

只检查变量是否存在，不打印值：

```bash
for name in OPENAI_BASE_URL OPENAI_API_KEY OPENAI_MODEL_ID; do
  if [ -n "${!name:-}" ]; then echo "$name=set"; else echo "$name=unset"; fi
done
```

## 3. 先运行两类代表问题

更新代码并确认回归测试：

```bash
git pull origin main
make setup-hybrid
make test
```

运行一个可回答的基础问题：

```bash
make answer \
  QUERY="单极格式下什么逻辑门实现乘法？" \
  ANSWER_MODE=fundamentals
```

再运行一个语料外问题：

```bash
make answer \
  QUERY="语料中有没有随机计算 Transformer 加速器的实测准确率与能效？" \
  ANSWER_MODE=paper_lookup
```

第一个结果应为 `status: answered` 或 `partial`，并在每个 claim 下包含真实的 `span_id`、`chunk_id`、论文别名、PDF 页码和由系统取回的原文摘录。`partial` 只保留有证据的 claim，并在 `missing_information` 中明确尚无证据的子问题。第二个结果应为 `status: abstained`。如果模型输出不存在的 span ID，系统会返回 `abstained`，并在 `reason` 中说明校验失败原因。

## 4. 运行 Phase 2 评测

先用一道可回答题和一道不可回答题验证完整链路与 API 费用：

```bash
.venv/bin/python scripts/evaluate_phase2.py \
  --scope all --ids A1 E1 \
  --output artifacts/phase2_smoke_eval.json
```

随后单独评测全部 8 道不可回答题：

```bash
make evaluate-phase2-unanswerable
```

确认拒答行为后，再运行 50 道可回答题：

先对检索失败较多的 9 题运行当前回归，观察 planner 与 repair trace：

```bash
.venv/bin/python scripts/evaluate_phase2.py \
  --scope answerable \
  --ids A6 A9 C2 C4 C9 C10 D1 D2 D8 \
  --output artifacts/phase2_v5_regression.json
```

确认 `response.requested_aliases`、`response.evidence_aliases` 和 `response.missing_requested_aliases` 符合预期后，再运行全量：

```bash
make evaluate-phase2-answerable
```

评测结束后先分析拒答来源，不要立即更换模型：

```bash
make analyze-phase2
cat artifacts/phase2_failure_analysis.md
```

`phase2-grounded-v5-validator-repair` 沿用可信 span 引用与 alias planner，并增加一次有界生成修复。别名只来自问题正文和语料 manifest；规划器不读取答案卷或 `gold_aliases`。旧版 `citation_quote_not_found_in_evidence` 仅用于分析历史报告。如果新版出现 `claim_referenced_unknown_span`，说明模型编造了 span ID；如果 `explicit_target_evidence_hit_all` 低，先检查候选池召回；如果证据目标全覆盖但引用目标未全覆盖，再检查 repair trace 和生成 Prompt。

若最终引用覆盖低于检索指标，运行跨阶段归因：

```bash
make attribute-phase2
cat artifacts/phase2_attribution.md
```

当前报告直接记录本次回答实际使用的 `evidence_aliases`，归因脚本优先使用该 trace，而不是拿 Phase 1 Top-10 近似上下文。`explicit_target_missing_from_evidence` 表示问题点名的论文没有进入最终证据；`generation_omitted_explicit_target` 表示经过修复后仍有目标论文未被引用。对没有显式论文目标的基础题，参考答案来源只用于计算 Hit-any 和平均 recall。

对于 `grounded_alternative_source`，归因报告会保留人工审核材料，但该项不阻塞 Phase 3：

```bash
make review-phase2-references PHASE2_REPORT=artifacts/phase2_answerable_eval.json
```

终端和 `artifacts/phase2_attribution.md` 会列出问题原文、答案卷的全部 gold aliases、模型的每条 claim，以及每条 claim 对应的 alias、span、页码和 chunk ID。人工确认替代引用直接支持 claim 后，它属于通过引用合同的 grounded answer；gold alias 命中率继续保留为 benchmark 诊断指标，不参与 Phase 2 release gate。

生成修复是一个最多两次调用的有界循环。第一次输出发生 JSON/Pydantic 校验错误，或漏引已经进入证据的显式目标时，系统会把错误类型交给模型修复一次。响应中的 `generation_attempts` 和 `repair_reason` 记录该过程；`generation_repair_success_rate` 衡量触发修复后最终得到完整或部分回答的比例。未知 span 仍由应用层 fail-closed。参考答案来源不会进入修复 Prompt，避免 gold 泄漏。

如果报告中的中文别名显示为 `ç»Ľčż°` 一类字符，先区分数据损坏与查看器编码错误。`make test` 会逐条校验生成 chunk 的 paper ID、alias 和 title 是否与 UTF-8 manifest 完全一致。测试通过时数据没有发生该类乱码转换，应把编辑器或终端文件编码切换为 UTF-8；测试失败时重新运行 `make ingest` 和 `make dense-index`，再重跑 Phase 1/2 报告。

评测逐题写入 JSON，中途发生网络错误时，已完成结果仍会保留。主要指标含义：

- `behavior_accuracy`：可回答题给出完整或部分证据回答、不可回答题拒答的比例；
- `full_answer_rate_on_answerable`、`partial_answer_rate_on_answerable`：区分完整回答和明确披露缺失项的部分回答；
- `abstention_rate_on_unanswerable`：八道语料外问题的拒答率；
- `explicit_target_evidence_hit_all`：问题中 `[论文别名]` 是否全部进入实际上下文；
- `explicit_target_citation_hit_all`、`mean_explicit_target_citation_recall`：显式目标是否全部被引用，以及平均覆盖比例；
- `reference_source_hit_any`、`mean_reference_source_recall`：最终引用对答案卷参考来源的命中与平均覆盖，作为弱监督质量信号；
- `legacy_reference_source_hit_all`：是否引用答案卷列出的全部来源，只保留作历史诊断，不作为 Phase 2 准入门槛；
- `grounded_response_rate_on_answerable`、`accepted_citation_contract_violations`：可回答题中满足 claim-level 引用合同的比例和违规数；
- `generation_repair_rate`、`generation_repair_success_rate`：生成修复触发比例与修复后的可用响应率；
- `mean_claims_when_responded`：完整和部分回答中的平均原子 claim 数；
- `mean_latency_ms`：检索、生成和校验的端到端平均延迟。

答案卷中的参考来源可能互相替代，也可能只是举例集合，因此不能把“全部参考来源都被引用”当成每题硬要求。显式目标来自用户问题，适合衡量跨论文任务是否完整执行；参考来源指标验证引用方向是否合理。两者都不等价于答案事实完全正确。Phase 4 会进一步加入原子答案评分。span ID 保证展示的摘录一定来自检索证据，也不能单独证明该摘录在语义上完整支持 claim；需要在失败分析中人工抽查。

## 5. 代码工作流

| 工作 | 代码入口 | 需要理解的原因 |
|---|---|---|
| API 适配 | `OpenAICompatibleChatClient` | 中转 API 和 DeepSeek 共用同一接口，密钥只从环境读取 |
| 模式判断 | `infer_mode` | 不同任务需要不同的回答组织约束 |
| 显式目标解析 | `extract_requested_aliases` | 只接受语料中存在的方括号别名，避免把 `[0, 1]` 当论文 |
| 定向补召回 | `BM25Index.search_alias`、`HybridRetriever.search_alias` | 处理 `Sobol`/`PSobol` 等名称碰撞，保证规划器有目标论文候选 |
| 证据规划 | `plan_evidence` | 跨论文问题先覆盖每个显式目标，再填充高分证据 |
| 上下文预算 | `select_evidence` | 在目标覆盖后控制 token、延迟和无关证据数量 |
| 防提示注入 | `SYSTEM_PROMPT`、`build_user_prompt` | 明确把论文文本当作不可信数据而非指令 |
| 输出合同 | `RawAnswer`、`RawClaim` | 禁止自由格式输出绕过引用检查 |
| 证据切片 | `split_evidence_spans` | 为原文分配受信任的 `E1.S1` 等 ID，避免模型复制原文造成格式误杀 |
| 引用验证 | `GroundedAnswerService.answer` | span ID 必须来自本次 Prompt，原文摘录由应用层取回 |
| 可信元数据绑定 | `Citation` | 页码、论文 ID 由应用生成，不能由模型编造 |
| 拒答 | `abstention` | 缺证据、坏 JSON和未知 span 都采用 fail-closed |
| 离线评测 | `scripts/evaluate_phase2.py` | gold 只在模型回答后用于评分，避免泄漏 |
| HTTP 接口 | `POST /answer` | 将同一服务提供给后续 UI 和部署阶段 |

核心文件为 `src/sc_rag/answering.py`；数据结构在 `src/sc_rag/models.py`；FastAPI 接口在 `src/sc_rag/api.py`。

## 6. FastAPI 调用

```bash
make api PORT=8002
```

在另一个终端调用：

```bash
curl -X POST http://127.0.0.1:8002/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"单极格式下什么逻辑门实现乘法？","mode":"fundamentals"}'
```

第一次请求需要加载本地 embedding 模型，明显慢于后续请求。`/docs` 可查看请求和响应 schema。

## 7. 本阶段理解门槛

进入下一小步前，应能解释：

1. 为什么检索命中正确论文仍不等于答案有依据；
2. 为什么页码和 chunk ID 必须由应用层绑定；
3. 为什么让模型引用可信 span ID 比要求它逐字复制 quote 更稳定，以及为什么 span 有效仍不等于语义蕴含；
4. 为什么不可回答题必须与可回答题分开评测；
5. 为什么 API 网络错误应抛出服务错误，而模型坏输出应安全拒答；
6. 上下文过多如何同时影响费用、延迟和忠实度。

最终必须使用完整的 50 道可回答题和 8 道不可回答题报告执行统一准入检查：

```bash
make assess-phase2
cat artifacts/phase2_readiness.md
```

只有 `passed: true` 才结束 Phase 2。硬门槛为：完整题集；可回答题 usable response rate 不低于 0.90；显式目标进入证据的 Hit-all 不低于 0.90；显式目标引用 Hit-all 不低于 0.85；8 道不可回答题拒答率不低于 0.875；被接受回答的引用合同违规数为 0。reference-source Hit-any、平均 reference recall 和经人工审核的替代来源数量只用于诊断，不阻塞进入 Phase 3。

