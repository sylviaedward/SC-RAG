# Phase 5：研究工作台 MVP

Phase 5 把已经通过评测的 FastAPI 能力包装成可演示的研究工作台。界面只负责交互和展示，检索、证据规划、生成、引用校验仍由后端完成。

## 启动

终端一：

```bash
make api PORT=8000
```

终端二：

```bash
make setup-ui
make ui
```

在 Codespaces 中打开转发的 8501 端口。若 FastAPI 不在 8000 端口，启动 UI 前设置：

```bash
export SC_RAG_API_URL="http://127.0.0.1:8001"
make ui
```

## 体验顺序

1. 用基础知识问题验证 grounded answer 和页码证据。
2. 用 `比较 [Sobol] 和 [P2LSG]` 验证多论文覆盖和 citation claims。
3. 用不可回答问题验证拒答状态。
4. 在 Literature Review 页输入主题和至少两个别名，观察逐篇抽取后的比较表与研究假设。

## 代码链路

```text
ui/app.py
  → POST /answer 或 POST /literature-review
  → FastAPI response_model
  → GroundedAnswerService / LiteratureReviewService
  → 页面展示 claims、excerpt、页码和研究想法
```

验收要求：三个演示场景都能完成，证据 excerpt 与页码可展开查看，API 错误能在页面显示，界面不读取或展示任何 API key。
