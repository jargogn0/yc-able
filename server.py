#!/usr/bin/env python3
"""
19Labs API Server
- POST /api/run          → start research (JSON body)
- GET  /api/run/{id}/stream → SSE stream of live logs
- GET  /api/run/{id}/status → current status + results
- GET  /api/run/{id}/deploy → download deploy.zip
- GET  /                 → serves 19labs-app.html
"""
import asyncio, glob, json, os, signal, shutil, tempfile, threading, time, uuid, zipfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
if ALLOWED_ORIGINS == ["*"]:
    ALLOWED_ORIGINS = ["*"]

ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "").strip()
CSV_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not ACCESS_PASSWORD:
        return await call_next(request)
    if request.url.path in ("/", "/health", "/favicon.ico"):
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {ACCESS_PASSWORD}":
        return await call_next(request)
    token = request.query_params.get("token", "")
    if token == ACCESS_PASSWORD:
        return await call_next(request)
    return JSONResponse(status_code=401, content={"error": "Unauthorized. Set your access token in Settings."})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

RUNS: dict[str, dict] = {}
APP_HTML = Path(__file__).parent / "19labs-app.html"

def _cleanup_old_workspaces():
    cutoff = time.time() - (4 * 3600)  # 4 hours
    for ws_path in glob.glob(str(Path(tempfile.gettempdir()) / "19labs_*")):
        try:
            p = Path(ws_path)
            if p.is_dir() and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

def _cleanup_loop():
    while True:
        time.sleep(1800)
        try:
            _cleanup_old_workspaces()
        except Exception:
            pass

threading.Thread(target=_cleanup_loop, daemon=True).start()
_cleanup_old_workspaces()

# ── SERVE APP ──────────────────────────────────────────────────
@app.get("/")
def serve_app():
    if APP_HTML.exists():
        return HTMLResponse(APP_HTML.read_text())
    return HTMLResponse("<h1>19Labs</h1><p>Place 19labs-app.html next to server.py</p>")

# ── START RUN (JSON body) ──────────────────────────────────────
class RunRequest(BaseModel):
    filename: str
    csv: str           # full CSV text
    hint: str = ""
    budget: int = 6
    reliability_mode: str = "balanced"
    api_key: str = ""
    provider: str = "claude"
    continuous: bool = False

class ValidateKeyRequest(BaseModel):
    api_key: str = ""
    provider: str = "claude"

class DiscoverRequest(BaseModel):
    filename: str
    csv: str
    hint: str = ""
    api_key: str = ""
    provider: str = "claude"

class ChatRequest(BaseModel):
    message: str
    api_key: str = ""
    provider: str = "claude"
    context: dict = {}

def _fallback_discovery(profile: dict, hint: str = ""):
    headers = profile.get("headers", [])
    numeric = profile.get("numeric", [])
    target_candidates = profile.get("target_candidates", [])
    target = target_candidates[0] if target_candidates else (numeric[-1] if numeric else (headers[-1] if headers else "target"))
    objective = {
        "domain": "General",
        "task": "Regression",
        "target": target,
        "metric": "rmse",
        "direction": "lower_is_better",
        "confidence": 0.45,
        "reasoning": "Fallback inference used because AI discovery was unavailable.",
        "good_enough": "",
        "raw": "fallback",
    }
    discovery = {
        "recommended_objective": f"Predict {target} from the available features.",
        "recommended_metric": "rmse",
        "clarifying_questions": [
            f"Is `{target}` the business target you want to optimize?",
            "Should we optimize for prediction accuracy or explainability?",
            "Do you want conservative baseline models first, or aggressive model search?",
        ],
        "experiment_directions": [
            "Start with linear/tree baseline to establish signal.",
            "Add robust preprocessing and leakage checks.",
            "Try gradient boosting if baseline underperforms.",
        ],
        "risks": [
            "Potential target leakage if ID-like columns are predictive.",
            "Small dataset size may overfit complex models.",
        ],
        "first_iteration_plan": "Confirm target/metric, run a strong baseline, inspect feature signal, then iterate.",
        "decision_tree": [
            {
                "id": "target_confirmation",
                "question": f"Should we use `{target}` as the prediction target?",
                "options": [
                    f"Yes, keep `{target}` as target.",
                    "No, I will adjust the objective text manually.",
                    "Use AI default target detection.",
                ],
            },
            {
                "id": "optimization_focus",
                "question": "What should we optimize first?",
                "options": [
                    "Highest predictive accuracy.",
                    "Balanced accuracy and stability.",
                    "Fast iteration speed.",
                ],
            },
            {
                "id": "search_intensity",
                "question": "How aggressive should model search be?",
                "options": [
                    "Conservative baseline first.",
                    "Balanced search strategy.",
                    "Aggressive search for best score.",
                ],
            },
        ],
    }
    return {
        "profile": profile,
        "objective": objective,
        "discovery": discovery,
        "raw": "fallback",
        "used_fallback": True,
        "warning": hint and "Used fallback discovery; user hint captured." or "Used fallback discovery; add API key for richer guidance.",
    }

@app.post("/api/run")
async def start_run(req: RunRequest):
    if len(req.csv.encode("utf-8")) > CSV_MAX_BYTES:
        raise HTTPException(413, f"CSV too large. Max 10 MB, got {len(req.csv.encode('utf-8')) / (1024*1024):.1f} MB.")

    run_id = str(uuid.uuid4())[:12]
    ws = Path(tempfile.mkdtemp(prefix=f"19labs_{run_id}_"))

    # Write CSV to disk
    csv_path = ws / req.filename
    csv_path.write_text(req.csv)

    cancel_event = threading.Event()
    RUNS[run_id] = dict(
        id=run_id, status="running", ws=str(ws),
        csv=str(csv_path), logs=[], result=None,
        started=time.time(), provider=req.provider,
        cancel_event=cancel_event,
    )

    def background():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from engine import run_research

        def cb(tag, msg):
            RUNS[run_id]["logs"].append({
                "tag": tag, "msg": msg, "ts": time.strftime("%H:%M:%S")
            })

        try:
            resolved_api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            cb("sys", f"API key received: {'YES' if resolved_api_key else 'NO'} (length={len(resolved_api_key)}) | Provider: {req.provider or 'claude'}")
            if not resolved_api_key:
                raise RuntimeError(
                    "No API key available. Provide one in Settings or set ANTHROPIC_API_KEY / OPENAI_API_KEY on the server."
                )
            result = run_research(
                str(csv_path),
                workspace=str(ws),
                budget=req.budget,
                reliability_mode=req.reliability_mode,
                user_hint=req.hint,
                api_key=resolved_api_key,
                log_callback=cb,
                continuous=req.continuous,
                cancel_event=cancel_event,
                provider=req.provider or "claude",
            )
            RUNS[run_id]["result"] = result
            if RUNS[run_id]["status"] == "stopping":
                RUNS[run_id]["status"] = "done"  # graceful stop completed
            elif RUNS[run_id]["status"] != "error":
                RUNS[run_id]["status"] = "done"
        except Exception as e:
            msg = str(e)
            if msg.strip() == "Connection error.":
                msg = (
                    "API connection error. Key was provided but request failed. "
                    "Check internet/VPN/firewall/proxy, then retry."
                )
            RUNS[run_id]["status"] = "error"
            RUNS[run_id]["error"] = msg
            cb("error", msg)

    threading.Thread(target=background, daemon=True).start()
    return {"run_id": run_id}

@app.post("/api/validate-key")
async def validate_key(req: ValidateKeyRequest):
    provider = (req.provider or "claude").lower()
    resolved_api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_api_key:
        return {"ok": False, "error": "No API key provided."}
    try:
        if provider == "openai":
            from openai import OpenAI
            c = OpenAI(api_key=resolved_api_key)
            c.chat.completions.create(model="gpt-4o", max_tokens=1, messages=[{"role":"user","content":"ping"}])
        else:
            from anthropic import Anthropic
            c = Anthropic(api_key=resolved_api_key)
            c.messages.create(model="claude-sonnet-4-6", max_tokens=1, system="Respond minimally.", messages=[{"role":"user","content":"ping"}])
        return {"ok": True}
    except Exception as e:
        msg = str(e)
        if msg.strip() == "Connection error.":
            msg = "API connection error during key validation. Check internet/VPN/firewall/proxy."
        return {"ok": False, "error": msg}

@app.post("/api/discover")
async def discover(req: DiscoverRequest):
    if len(req.csv.encode("utf-8")) > CSV_MAX_BYTES:
        raise HTTPException(413, f"CSV too large. Max 10 MB, got {len(req.csv.encode('utf-8')) / (1024*1024):.1f} MB.")
    ws = Path(tempfile.mkdtemp(prefix="19labs_discover_"))
    csv_path = ws / req.filename
    csv_path.write_text(req.csv)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from engine import discover_user_need, profile_dataset
        profile = profile_dataset(str(csv_path))
        resolved_api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        if not resolved_api_key:
            fallback = _fallback_discovery(profile, req.hint)
            fallback["ok"] = True
            fallback["provider_note"] = "Using smart fallback discovery. Add an API key for richer AI analysis."
            return fallback

        result = discover_user_need(str(csv_path), user_hint=req.hint, api_key=resolved_api_key, provider=req.provider or "claude")
        result["used_fallback"] = False
        return {"ok": True, **result}
    except Exception as e:
        msg = str(e)
        if msg.strip() == "Connection error.":
            msg = "API connection error during discovery. Check internet/VPN/firewall/proxy."
        return {"ok": False, "error": msg, "used_fallback": False}
    finally:
        try:
            shutil.rmtree(ws, ignore_errors=True)
        except Exception:
            pass

# ── PUSH CODE TO WORKSPACE ─────────────────────────────────────
class PushCodeRequest(BaseModel):
    code: str

@app.post("/api/run/{run_id}/push-code")
async def push_code(run_id: str, req: PushCodeRequest):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    run = RUNS[run_id]
    ws = Path(run.get("ws", ""))
    if not ws.exists():
        raise HTTPException(404, "Workspace not found")
    train_py = ws / "train.py"
    train_py.write_text(req.code)
    return {"ok": True, "msg": "train.py updated in workspace"}

# ── CANCEL RUN ─────────────────────────────────────────────────
@app.post("/api/run/{run_id}/cancel")
async def cancel_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    run = RUNS[run_id]
    if run["status"] != "running":
        return {"ok": True, "msg": "Run already finished"}
    cancel_ev = run.get("cancel_event")
    if cancel_ev:
        cancel_ev.set()
    run["status"] = "stopping"
    run["logs"].append({"tag": "sys", "msg": "Stopping after current experiment... Results will be preserved.", "ts": time.strftime("%H:%M:%S")})
    return {"ok": True, "msg": "Stopping gracefully — results from completed experiments will be kept"}

# ── LIST RUNS ──────────────────────────────────────────────────
@app.get("/api/runs")
def list_runs():
    out = []
    for rid, run in sorted(RUNS.items(), key=lambda x: x[1].get("started", 0), reverse=True):
        out.append({
            "id": rid,
            "status": run["status"],
            "started": run.get("started"),
            "filename": run.get("csv", "").split("/")[-1] if run.get("csv") else "",
            "provider": run.get("provider", "claude"),
        })
    return {"runs": out[:50]}

# ── SSE STREAM ─────────────────────────────────────────────────
@app.get("/api/run/{run_id}/stream")
async def stream_logs(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")

    async def generate():
        sent = 0
        run_start = RUNS[run_id].get("started", time.time())
        while True:
            run = RUNS[run_id]
            logs = run["logs"]
            while sent < len(logs):
                entry = dict(logs[sent])
                entry["elapsed"] = round(time.time() - run_start, 1)
                yield f"data: {json.dumps(entry)}\n\n"
                sent += 1
            if run["status"] in ("done", "error"):
                yield f"data: {json.dumps({'tag':'sys','msg':'__DONE__','ts':time.strftime('%H:%M:%S'),'elapsed':round(time.time()-run_start,1)})}\n\n"
                break
            if run["status"] == "stopping":
                for _ in range(30):
                    await asyncio.sleep(1)
                    # Flush any new logs while waiting
                    while sent < len(run["logs"]):
                        entry = dict(run["logs"][sent])
                        entry["elapsed"] = round(time.time() - run_start, 1)
                        yield f"data: {json.dumps(entry)}\n\n"
                        sent += 1
                    if run["status"] in ("done", "error"):
                        break
                yield f"data: {json.dumps({'tag':'sys','msg':'__DONE__','ts':time.strftime('%H:%M:%S'),'elapsed':round(time.time()-run_start,1)})}\n\n"
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── STATUS ─────────────────────────────────────────────────────
@app.get("/api/run/{run_id}/status")
def get_status(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    run = RUNS[run_id]
    result = run.get("result")
    elapsed = round(time.time() - run.get("started", time.time()), 1)
    out = dict(status=run["status"], log_count=len(run["logs"]), elapsed=elapsed)
    if result:
        out["best"]      = result.get("best")
        out["objective"] = result.get("objective")
        out["history"]   = result.get("history")
        out["report"]    = result.get("report")
        out["deploy_path"] = result.get("deploy_path")
        out["diagnostics"] = result.get("diagnostics")
        out["executive_brief"] = result.get("executive_brief")
        out["artifacts"] = result.get("artifacts")
        out["total_experiments"] = result.get("total_experiments", len(result.get("history", [])))
        out["continuous_mode"] = result.get("continuous_mode", False)
        out["token_usage"] = result.get("token_usage", {})
    if run.get("error"):
        out["error"] = run["error"]
    return out


def _build_handoff_payload(run_id: str, result: dict):
    best = result.get("best") or {}
    obj = result.get("objective") or {}
    diagnostics = result.get("diagnostics") or {}
    brief = result.get("executive_brief") or diagnostics.get("executive_brief") or ""
    artifacts = result.get("artifacts") or {}
    artifact_urls = {
        k: f"/api/run/{run_id}/artifact/{k}"
        for k, v in artifacts.items()
        if v
    }
    metric_name = best.get("metric_name") or obj.get("metric", "metric")
    metric_val = best.get("metric_val")
    metric_text = "n/a" if metric_val is None else f"{float(metric_val):.6f}"
    handoff_md = (
        "# 19Labs Project Handoff\n\n"
        "## Outcome\n"
        f"- Best model: **{best.get('model', 'N/A')}**\n"
        f"- Best metric: **{metric_name} = {metric_text}**\n"
        f"- Run quality score: **{diagnostics.get('yc_readiness_score', 'N/A')}/100 ({diagnostics.get('yc_grade', 'N/A')})**\n"
        f"- Headline: {diagnostics.get('headline', 'N/A')}\n\n"
        "## Objective\n"
        f"- Task: {obj.get('task', 'N/A')}\n"
        f"- Target: {obj.get('target', 'N/A')}\n"
        f"- Reliability mode: {obj.get('reliability_mode', diagnostics.get('reliability_mode', 'balanced'))}\n\n"
        "## Executive brief\n"
        f"{brief or 'N/A'}\n\n"
        "## Links\n"
        f"- Deploy package: /api/run/{run_id}/deploy\n"
        + "\n".join([f"- {k}: {u}" for k, u in artifact_urls.items()])
    )
    return {
        "run_id": run_id,
        "status": "done",
        "best": best,
        "objective": obj,
        "diagnostics": diagnostics,
        "executive_brief": brief,
        "deploy_url": f"/api/run/{run_id}/deploy",
        "artifact_urls": artifact_urls,
        "handoff_markdown": handoff_md,
    }


@app.get("/api/run/{run_id}/handoff")
def get_handoff(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    run = RUNS[run_id]
    if run.get("status") != "done":
        raise HTTPException(409, "Run is not completed yet")
    result = run.get("result") or {}
    return _build_handoff_payload(run_id, result)


@app.get("/api/run/{run_id}/project-handoff")
def get_project_handoff(run_id: str):
    return get_handoff(run_id)

# ── DOWNLOAD DEPLOY ZIP ────────────────────────────────────────
@app.get("/api/run/{run_id}/deploy")
def download_deploy(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    result = RUNS[run_id].get("result")
    if not result or not result.get("deploy_path"):
        raise HTTPException(404, "Deploy package not ready yet")
    dp = Path(result["deploy_path"])
    if not dp.exists():
        raise HTTPException(404, "Deploy directory missing")
    zip_path = dp.parent / f"deploy_{run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in dp.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(dp))
    return FileResponse(
        str(zip_path),
        filename=f"19labs_model_{run_id}.zip",
        media_type="application/zip",
    )


@app.get("/api/run/{run_id}/project-pack")
def download_project_pack(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    run = RUNS[run_id]
    result = run.get("result") or {}
    ws = Path(run.get("ws", ""))
    if not ws.exists():
        raise HTTPException(404, "Run workspace not available")

    artifacts = result.get("artifacts") or {}
    include_paths: list[tuple[Path, Path]] = []

    # Core run files from workspace
    for name in ["objective.json", "profile.json", "program.md", "train.py", "results.tsv", "final_report.md", "requirements.txt"]:
        p = ws / name
        if p.exists() and p.is_file():
            include_paths.append((p, Path(name)))

    # Dynamic artifacts
    for key in ["prepare_py", "analysis_ipynb", "progress_png", "train_test_png", "model_comparison_png", "metrics_overview_png", "experiment_timeline_png", "results_tsv", "train_py", "program_md", "final_report_md"]:
        p = artifacts.get(key)
        if p:
            fp = Path(p)
            if fp.exists() and fp.is_file():
                include_paths.append((fp, Path(fp.name)))

    # Include deploy bundle contents if available
    dp = result.get("deploy_path")
    if dp:
        dpp = Path(dp)
        if dpp.exists() and dpp.is_dir():
            for f in dpp.rglob("*"):
                if f.is_file():
                    include_paths.append((f, Path("deploy") / f.relative_to(dpp)))

    # Add concise handoff file directly in project pack
    handoff_payload = _build_handoff_payload(run_id, result)
    handoff_text = handoff_payload.get("handoff_markdown", "")
    tmp_handoff = ws / "PROJECT_HANDOFF.md"
    tmp_handoff.write_text(handoff_text)
    include_paths.append((tmp_handoff, Path("PROJECT_HANDOFF.md")))

    zip_path = ws / f"project_pack_{run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        seen = set()
        for src, arc in include_paths:
            arc_name = str(arc)
            if arc_name in seen:
                continue
            seen.add(arc_name)
            z.write(src, arc_name)

    return FileResponse(
        str(zip_path),
        filename=f"19labs_project_{run_id}.zip",
        media_type="application/zip",
    )

@app.get("/api/run/{run_id}/artifact/{artifact_name}")
def download_artifact(run_id: str, artifact_name: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    result = RUNS[run_id].get("result") or {}
    artifacts = result.get("artifacts") or {}
    allowed = {
        "program_md": "program.md",
        "prepare_py": "prepare.py",
        "analysis_ipynb": "analysis.ipynb",
        "progress_png": "progress.png",
        "train_test_png": "train_test.png",
        "model_comparison_png": "model_comparison.png",
        "metrics_overview_png": "metrics_overview.png",
        "experiment_timeline_png": "experiment_timeline.png",
        "results_tsv": "results.tsv",
        "train_py": "train.py",
        "final_report_md": "final_report.md",
    }
    # Also allow per-experiment plots like exp_01_predictions_png
    if artifact_name not in allowed and not artifact_name.startswith("exp_"):
        raise HTTPException(404, "Unknown artifact")
    p = artifacts.get(artifact_name)
    if not p:
        run = RUNS[run_id]
        ws = Path(run.get("ws", ""))
        fallback_name = allowed.get(artifact_name, "")
        fallback_path = ws / fallback_name if ws.exists() and fallback_name else None
        if fallback_path and fallback_path.exists():
            p = str(fallback_path)
        else:
            raise HTTPException(404, "Artifact not available")
    fp = Path(p)
    if not fp.exists() or not fp.is_file():
        raise HTTPException(404, "Artifact file missing")
    media = "application/octet-stream"
    if artifact_name == "analysis_ipynb":
        media = "application/x-ipynb+json"
    elif artifact_name.endswith("_png"):
        media = "image/png"
    elif artifact_name.endswith("_md"):
        media = "text/markdown"
    elif artifact_name.endswith("_py"):
        media = "text/x-python"
    elif artifact_name.endswith("_tsv"):
        media = "text/tab-separated-values"
    return FileResponse(str(fp), filename=allowed[artifact_name], media_type=media)

# ── TRAIN.PY CONTENT (text endpoint) ──────────────────────────
@app.get("/api/run/{run_id}/train-py")
def get_train_py(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    ws = Path(RUNS[run_id].get("ws", ""))
    tp = ws / "train.py"
    if tp.exists():
        return {"ok": True, "code": tp.read_text()}
    return {"ok": False, "code": ""}

# ── PACKAGES ───────────────────────────────────────────────────
_cached_packages: list[str] | None = None

@app.get("/api/packages")
def get_packages():
    global _cached_packages
    if _cached_packages is None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from engine import detect_packages
        _cached_packages = detect_packages()
    return {"available": _cached_packages}

# ── CHAT ───────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(400, "API key required for chat")
    from engine import chat_with_data
    try:
        response = chat_with_data(req.message, req.context, api_key, req.provider)
        return {"ok": True, "response": response}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── CREATE INFERENCE API ────────────────────────────────────────
@app.post("/api/run/{run_id}/create-api")
async def create_api_server(run_id: str, request: Request):
    body = await request.json()
    api_key = body.get("api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    provider = body.get("provider", "claude")
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    run = RUNS[run_id]
    result = run.get("result")
    if not result:
        raise HTTPException(400, "Run not finished yet")
    ws = Path(run.get("ws", ""))
    deploy_path = result.get("deploy_path")
    dp = Path(deploy_path) if deploy_path and Path(deploy_path).exists() else ws
    train_py = ""
    for candidate in [dp / "train.py", ws / "train.py"]:
        if candidate.exists():
            train_py = candidate.read_text()
            break
    best = result.get("best") or {}
    obj = result.get("objective") or {}
    from engine import generate_inference_server
    generated = generate_inference_server(train_py, best, obj, api_key, provider)
    if "error" in generated and not generated.get("inference_server_py"):
        raise HTTPException(500, f"Generation failed: {generated['error']}")
    api_dir = ws / "api_server"
    api_dir.mkdir(exist_ok=True)
    inference_py = generated.get("inference_server_py", "")
    requirements = generated.get("requirements_txt", "fastapi\nuvicorn\njoblib\nnumpy\npandas\nscikit-learn\n")
    dockerfile = generated.get("dockerfile", "FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\nRUN pip install --no-cache-dir -r requirements.txt\nCMD uvicorn inference_server:app --host 0.0.0.0 --port $PORT\n")
    (api_dir / "inference_server.py").write_text(inference_py)
    (api_dir / "requirements.txt").write_text(requirements)
    (api_dir / "Dockerfile").write_text(dockerfile)
    (api_dir / "railway.json").write_text(json.dumps({
        "$schema": "https://railway.app/railway.schema.json",
        "deploy": {"startCommand": "uvicorn inference_server:app --host 0.0.0.0 --port $PORT", "healthcheckPath": "/health"}
    }, indent=2))
    model_name = best.get("model", "MLModel")
    metric_name = best.get("metric_name", "metric")
    metric_val = best.get("metric_val", "N/A")
    (api_dir / "README.md").write_text(
        f"# {model_name} Inference API\n\nGenerated by 19Labs.\n\n"
        f"## Model\n- Model: **{model_name}**\n- Task: {obj.get('task','N/A')}\n"
        f"- Target: `{obj.get('target','N/A')}`\n- {metric_name}: `{metric_val}`\n\n"
        f"## Deploy to Railway\n```bash\nrailway login && railway up\n```\n\n"
        f"## Run locally\n```bash\npip install -r requirements.txt\nuvicorn inference_server:app --reload\n```\n\n"
        f"## Predict\n```bash\ncurl -X POST http://localhost:8000/predict \\\n"
        f"  -H 'Content-Type: application/json' \\\n  -d '{{\"feature1\": 1.0, \"feature2\": \"value\"}}'\n```\n"
    )
    # Copy model files from workspace
    for pattern in ["*.pkl", "*.joblib", "*.pt", "*.h5", "best_model*", "model.*"]:
        for f in list(dp.glob(pattern)) + list(ws.glob(pattern)):
            if f.is_file() and f.parent != api_dir:
                shutil.copy2(f, api_dir / f.name)
    zip_path = ws / f"api_server_{run_id[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in api_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(api_dir))
    return FileResponse(str(zip_path), filename=f"19labs_api_{run_id[:8]}.zip", media_type="application/zip")

# ── HEALTH ─────────────────────────────────────────────────────
@app.get("/favicon.ico")
def favicon():
    return HTMLResponse(status_code=204)

@app.get("/health")
def health():
    return {"status": "ok", "runs": len(RUNS)}

@app.get("/api/provider-status")
def provider_status():
    has_server_key = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    return {"provider": "claude", "server_key_available": has_server_key, "cached_key_available": False}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8019))
    print(f"\n  19Labs Server → http://localhost:{port}")
    print(f"  Open browser: http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
