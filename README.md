# 19Labs

Autonomous ML research engine for tabular datasets.

Upload a CSV, let the engine infer the task, generate training experiments, execute them, iterate on real metrics, and package the best model for deployment.

## What is in this repo

- `server.py`: FastAPI server and API endpoints.
- `engine.py`: iterative research loop (profile -> infer -> write -> execute -> improve).
- `19labs-app.html`: single-page frontend UI.
- `autoresearch-master/`: separate LLM autoresearch project.

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
# then set ANTHROPIC_API_KEY in .env
```

3. Run the app:

```bash
python server.py
```

Open [http://localhost:8019](http://localhost:8019).

## API overview

- `POST /api/run` start a research run (CSV text + budget + API key)
- `GET /api/run/{id}/stream` live SSE logs
- `GET /api/run/{id}/status` run status and final result
- `GET /api/run/{id}/deploy` download deploy artifact zip
- `GET /api/run/{id}/project-pack` download full project bundle zip
- `GET /api/run/{id}/artifact/{artifact_name}` download generated artifact
- `GET /api/run/{id}/project-handoff` generate project handoff payload
- `GET /health` health check

## Notes

- Current backend execution path uses Anthropic via `engine.py`.
- The UI shows multiple providers, but only Claude is implemented server-side today.
- Run data is in-memory and temporary; restart clears active runs.

## MCP + Skills (project setup)

This repo now includes a project-level MCP server and Cursor skills:

- MCP config: `.cursor/mcp.json`
- MCP server: `mcp/19labs_mcp_server.py`
- Skills:
  - `.cursor/skills/19labs-data-scientist/SKILL.md`
  - `.cursor/skills/19labs-experiment-audit/SKILL.md`

### MCP quick start

1. Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

This installs `fastmcp`, used by the project MCP server.

2. Run 19Labs backend:

```bash
python server.py
```

3. Cursor should auto-discover project MCP from `.cursor/mcp.json`.

### MCP tools exposed

- `health`
- `profile_csv`
- `discover_direction`
- `start_research_run`
- `get_run_status`
- `get_artifact_url`
- `bootstrap_project` (one-command profile + discovery + optional launch/poll)
- `bootstrap_and_export_handoff` (one-command run + investor-ready handoff payload)
