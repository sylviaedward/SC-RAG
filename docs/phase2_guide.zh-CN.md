# Phase 2：证据约束回答操作与代码导读

Phase 2 的目标不是“让 LLM 看起来会回答”，而是让系统只交付能追溯到检索 chunk 和 PDF 页码的 claim，并在证据不足、引用伪造或输出损坏时拒答。

## 1. 当前最小闭环

```text
问题 + 回答模式
  -> Hybrid RRF 检索 Top-10
  -> 上下文预算裁剪（最多 10 chunks / 14000 字符）
  -> 将论文内容标记为不可信数据
  -> OpenAI-compatible Chat Completions
  -> 严格 JSON 解析
  -> evidence ID 白名单校验
  -> 支持性原文摘录校验
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

第一个结果应为 `status: answered` 或 `partial`，并在每个 claim 下包含真实的 `chunk_id`、论文别名和 PDF 页码。`partial` 只保留有证据的 claim，并在 `missing_information` 中明确尚无证据的子问题。第二个结果应为 `status: abstained`。如果模型输出了不存在的 evidence ID，或给出的 quote 无法在对应 chunk 中找到，系统也会返回 `abstained`，并在 `reason` 中说明校验失败原因。

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

```bash
make evaluate-phase2-answerable
```

评测结束后先分析拒答来源，不要立即更换模型：

```bash
make analyze-phase2
cat artifacts/phase2_failure_analysis.md
```

如果 `citation_quote_not_found_in_evidence` 占多数，应调整 quote 生成或规范化；如果 `model_declared_insufficient` 占多数，应检查上下文覆盖和拒答提示；如果结构校验错误占多数，才优先考虑模型的指令遵循能力。报告还会给出只在已回答样本上计算的 conditional citation gold Hit-any/Hit-all，避免把安全拒答和引用精度混成一个指标。

评测逐题写入 JSON，中途发生网络错误时，已完成结果仍会保留。主要指标含义：

- `behavior_accuracy`：可回答题给出完整或部分证据回答、不可回答题拒答的比例；
- `full_answer_rate_on_answerable`、`partial_answer_rate_on_answerable`：区分完整回答和明确披露缺失项的部分回答；
- `abstention_rate_on_unanswerable`：八道语料外问题的拒答率；
- `citation_gold_hit_any/all`：最终 claim 引用是否覆盖 gold 论文；
- `mean_claims_when_responded`：完整和部分回答中的平均原子 claim 数；
- `mean_latency_ms`：检索、生成和校验的端到端平均延迟。

这里的 gold alias 指标验证引用论文是否正确，不等价于答案事实完全正确。Phase 4 会进一步加入原子答案评分。当前的 quote 校验确保引用文本真实存在，也不能单独证明摘录在语义上完整支持 claim；需要在失败分析中人工抽查。

## 5. 代码工作流

| 工作 | 代码入口 | 需要理解的原因 |
|---|---|---|
| API 适配 | `OpenAICompatibleChatClient` | 中转 API 和 DeepSeek 共用同一接口，密钥只从环境读取 |
| 模式判断 | `infer_mode` | 不同任务需要不同的回答组织约束 |
| 上下文预算 | `select_evidence` | 控制 token、延迟和无关证据数量 |
| 防提示注入 | `SYSTEM_PROMPT`、`build_user_prompt` | 明确把论文文本当作不可信数据而非指令 |
| 输出合同 | `RawAnswer`、`RawClaim` | 禁止自由格式输出绕过引用检查 |
| 引用验证 | `GroundedAnswerService.answer` | evidence ID 必须来自本次检索，quote 必须存在于该 chunk |
| 可信元数据绑定 | `Citation` | 页码、论文 ID 由应用生成，不能由模型编造 |
| 拒答 | `abstention` | 缺证据、坏 JSON、未知引用、伪造 quote 都采用 fail-closed |
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
3. 为什么 quote 存在性校验比只校验 `E1` 更强，但仍不等于语义蕴含；
4. 为什么不可回答题必须与可回答题分开评测；
5. 为什么 API 网络错误应抛出服务错误，而模型坏输出应安全拒答；
6. 上下文过多如何同时影响费用、延迟和忠实度。

