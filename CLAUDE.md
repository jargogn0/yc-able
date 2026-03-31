# 19Labs — Claude Code Development Guide

## Project Overview

**19Labs** is a no-code AutoML web app. Drop a CSV → auto-profile → auto-train (AutoGluon) → predict on new data. Deployed on Railway.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python · FastAPI · Uvicorn |
| **ML Engine** | AutoGluon TabularPredictor (exp-1 baseline) · LightGBM · XGBoost · CatBoost |
| **AI Orchestration** | Anthropic Claude (claude-3-5-haiku / claude-3-5-sonnet) |
| **Frontend** | Vanilla JS · Single HTML file (`19labs-app.html`) |
| **Persistence** | SQLite (`runs.db`) · filesystem model store |
| **MCP Server** | `mcp/19labs_mcp_server.py` (FastMCP) |
| **Deployment** | Railway (Nixpacks) |

## Key Files

- `engine.py` — AutoML research engine: profile → discover → write train.py → execute → iterate
- `server.py` — FastAPI server: upload CSV, run experiments, serve predictions
- `19labs-app.html` — Full frontend (single file, no build step)
- `mcp/19labs_mcp_server.py` — MCP server exposing project tools to Claude Code / Cursor

## MCP Server

The project ships an MCP server for data-science workflows. Start it with:

```bash
python3 mcp/19labs_mcp_server.py
```

Or register with Claude Code / Cursor via `.mcp.json` (already configured in repo root).

**Available tools:** `health`, `profile_csv`, `discover_direction`, `start_research_run`, `get_run_status`, `get_artifact_url`, `bootstrap_project`, `bootstrap_and_export_handoff`

## Development Patterns

### Agent Flow
```
CSV upload → profile_dataset() → analyze_domain() [structured JSON]
→ infer_objective() → discover_user_need() → show summary in chat
→ user says "go" → run_experiments() → AutoGluon exp-1 → iterate
```

### Experiment Tiers
- **Exp 1**: AutoGluon `best_quality` preset — handles ANY tabular dataset automatically
- **Exp 2**: LLM-written code using `better_approach` from domain analysis
- **Exp 3+**: LLM-written code using `advanced_approach` (stacking, etc.)

### Model Persistence
- AutoGluon saves to `ag_models/` in workspace
- Server copies `ag_models/` → `_MODELS_DIR/{run_id}_ag/` for persistence across restarts
- `model.pkl` stores `{"type": "autogluon", "ag_path": "..."}` reference

### Frontend Data Flow
- `runDiscover()` — POST `/api/discover` → parse structured profile → render summary HTML
- User says "go" → POST `/api/run` → SSE stream `/api/run/{id}/stream` → progress updates
- `predictFile()` — POST `/api/predict-file` → AutoGluon or sklearn pkl chain

## Code Conventions

- **engine.py**: All LLM calls go through `_llm()`. Prompts forbid questions when data context available.
- **server.py**: Heavy lifting (subprocess, file I/O) is async-safe via `asyncio.to_thread`.
- **HTML**: `safeHtml()` escapes XSS, `esc()` for raw text, `md()` for markdown rendering.
- **Signals**: Always parse raw signal strings to human language before displaying in chat.

## Agent Operating Principles

Keep changes small and targeted. Read existing patterns before modifying. Validate with `python -c "import engine; import server"` for quick syntax checks. For the HTML file, check `safeHtml()` and signal parsing when touching chat rendering code.
