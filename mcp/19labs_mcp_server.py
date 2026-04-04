#!/usr/bin/env python3
"""
19Labs MCP server for data-science workflows.

Exposes project-aware tools so Cursor/agents can:
- profile datasets
- run discovery
- start/poll research runs
- fetch run artifacts
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict
from urllib import error, request

import pandas as pd
try:
    from fastmcp import FastMCP
except Exception as e:
    raise RuntimeError(
        "fastmcp is required for the 19Labs MCP server. Install with: pip install -r requirements.txt"
    ) from e

API_BASE = os.environ.get("NINETEENLABS_API_BASE", "http://localhost:8019").rstrip("/")
DEFAULT_PROVIDER = os.environ.get("NINETEENLABS_PROVIDER", "claude")
DEFAULT_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

mcp = FastMCP("19labs-datascientist")

_VALID_RELIABILITY_MODES = {"conservative", "balanced", "aggressive"}
_VALID_PROVIDERS = {"claude", "openai", "bedrock"}
_VALID_ARTIFACTS = {
    "program_md", "prepare_py", "analysis_ipynb",
    "progress_png", "results_tsv", "train_py", "final_report_md",
}


def _is_local() -> bool:
    """Return True if API_BASE points to localhost — enables direct file-path passing."""
    return any(h in API_BASE for h in ("localhost", "127.0.0.1", "0.0.0.0"))


def _validate_csv_path(csv_path: str) -> Path:
    p = Path(csv_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise ValueError(f"CSV file not found: {p}")
    if p.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        raise ValueError(f"Expected a CSV file, got: {p.suffix}")
    return p


def _validate_reliability_mode(mode: str) -> str:
    if mode not in _VALID_RELIABILITY_MODES:
        raise ValueError(f"reliability_mode must be one of {_VALID_RELIABILITY_MODES}, got: {mode!r}")
    return mode


def _validate_budget(budget: int) -> int:
    if not (1 <= budget <= 20):
        raise ValueError(f"budget must be between 1 and 20, got: {budget}")
    return budget


def _check_api_key(provider: str = "claude") -> None:
    """Raise if the required API key is missing for the given provider."""
    if provider in ("claude", "anthropic"):
        key = DEFAULT_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it before using 19Labs MCP tools:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )
    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before using 19Labs MCP tools:\n"
                "  export OPENAI_API_KEY=sk-..."
            )


def _http_json(method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = f"{API_BASE}{path}"
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, method=method.upper(), data=body, headers=headers)
    try:
        with request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return {"_empty_response": True}
            return json.loads(raw)
    except error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        msg = raw or str(e)
        raise RuntimeError(f"{method} {path} → HTTP {e.code}: {msg}") from e
    except error.URLError as e:
        raise RuntimeError(
            f"Cannot reach 19Labs API at {API_BASE}: {e}\n"
            "Make sure the server is running: uvicorn server:app --port 8019"
        ) from e


def _csv_payload_for(p: Path, hint: str = "", provider: str = DEFAULT_PROVIDER) -> Dict[str, Any]:
    """
    Build the CSV portion of a discover/run payload.
    - Local API: send csv_file_path so the server reads it directly (no OOM risk).
    - Remote API: send a truncated sample (first 500 rows) to keep payloads small.
    """
    if _is_local():
        return {
            "filename": p.name,
            "csv": "",
            "csv_file_path": str(p),
        }
    # Remote — send truncated sample
    try:
        df_sample = pd.read_csv(p, nrows=500)
        csv_text = df_sample.to_csv(index=False)
    except Exception:
        csv_text = p.read_text(encoding="utf-8", errors="replace")[:400_000]
    return {
        "filename": p.name,
        "csv": csv_text,
    }


def _artifact_urls(run_id: str, artifacts: Dict[str, Any] | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not artifacts:
        return out
    for key, value in artifacts.items():
        if value:
            out[key] = f"{API_BASE}/api/run/{run_id}/artifact/{key}"
    return out


def _discovery_summary(discovery_payload: Dict[str, Any]) -> Dict[str, Any]:
    d = (discovery_payload or {}).get("discovery") or {}
    o = (discovery_payload or {}).get("objective") or {}
    return {
        "task": o.get("task"),
        "target": o.get("target"),
        "metric": o.get("metric"),
        "recommended_objective": d.get("recommended_objective"),
        "recommended_metric": d.get("recommended_metric"),
        "clarifying_questions": (d.get("clarifying_questions") or [])[:5],
        "directions": (d.get("experiment_directions") or [])[:5],
        "csv_cache_id": discovery_payload.get("csv_cache_id") or "",
    }


def _handoff_markdown(
    project_name: str,
    csv_path: str,
    status: Dict[str, Any],
    run_id: str | None,
    artifact_urls: Dict[str, str],
) -> str:
    best = status.get("best") or {}
    diag = status.get("diagnostics") or {}
    obj = status.get("objective") or {}
    brief = status.get("executive_brief") or diag.get("executive_brief") or ""
    model = best.get("model", "N/A")
    metric_name = best.get("metric_name", obj.get("metric", "metric"))
    metric_val = best.get("metric_val")
    metric_text = "N/A" if metric_val is None else f"{float(metric_val):.6f}"
    score = diag.get("yc_readiness_score", "N/A")
    grade = diag.get("yc_grade", "N/A")
    headline = diag.get("headline", "")
    mode = (diag.get("reliability_mode") or obj.get("reliability_mode") or "balanced")
    deploy_url = f"{API_BASE}/api/run/{run_id}/deploy" if run_id else ""
    links = []
    for k in ["program_md", "prepare_py", "analysis_ipynb", "progress_png", "results_tsv", "train_py", "final_report_md"]:
        u = artifact_urls.get(k)
        if u:
            links.append(f"- `{k}`: {u}")
    links_txt = "\n".join(links) if links else "- (artifact links pending)"
    return (
        f"# 19Labs Investor Handoff\n\n"
        f"## Project\n"
        f"- Name: {project_name}\n"
        f"- Dataset: {csv_path}\n"
        f"- Run ID: {run_id or 'N/A'}\n"
        f"- Reliability mode: {mode}\n\n"
        f"## Outcome\n"
        f"- Best model: **{model}**\n"
        f"- Best metric: **{metric_name} = {metric_text}**\n"
        f"- YC readiness: **{score}/100 ({grade})**\n"
        f"- Headline: {headline or 'N/A'}\n\n"
        f"## Executive brief\n"
        f"{brief or 'N/A'}\n\n"
        f"## Downloads\n"
        f"- Deploy package: {deploy_url or 'N/A'}\n"
        f"{links_txt}\n"
    )


# ── TOOLS ───────────────────────────────────────────────────────────────────

@mcp.tool()
def health() -> Dict[str, Any]:
    """Check if local 19Labs API is reachable."""
    return _http_json("GET", "/health")


@mcp.tool()
def profile_csv(csv_path: str) -> Dict[str, Any]:
    """Lightweight local dataset profile for fast triage."""
    p = _validate_csv_path(csv_path)
    df = pd.read_csv(p)
    null_ratio = {c: float(df[c].isna().mean()) for c in df.columns}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return {
        "path": str(p),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "top_null_columns": sorted(null_ratio.items(), key=lambda x: x[1], reverse=True)[:8],
    }


@mcp.tool()
def discover_direction(
    csv_path: str,
    hint: str = "",
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    """
    Run 19Labs discovery to get objective suggestions and dataset analysis.

    Returns a summary including csv_cache_id — pass this to start_research_run
    to avoid re-uploading the file.
    """
    _check_api_key(provider)
    p = _validate_csv_path(csv_path)
    payload = {
        **_csv_payload_for(p, hint=hint, provider=provider),
        "hint": hint,
        "provider": provider,
        "api_key": DEFAULT_KEY,
    }
    result = _http_json("POST", "/api/discover", payload)
    # Surface cache_id at top level for easy reuse
    if "csv_cache_id" not in result:
        result["csv_cache_id"] = ""
    return result


@mcp.tool()
def start_research_run(
    csv_path: str,
    objective_hint: str = "",
    budget: int = 6,
    reliability_mode: str = "balanced",
    provider: str = DEFAULT_PROVIDER,
    csv_cache_id: str = "",
) -> Dict[str, Any]:
    """
    Start a full research run and return run_id immediately (non-blocking).

    Pass csv_cache_id from a prior discover_direction call to skip re-uploading.
    Use get_run_status(run_id) to poll progress.
    """
    _check_api_key(provider)
    p = _validate_csv_path(csv_path)
    _validate_budget(budget)
    _validate_reliability_mode(reliability_mode)
    if csv_cache_id:
        csv_part = {"filename": p.name, "csv": "", "csv_cache_id": csv_cache_id}
    else:
        csv_part = _csv_payload_for(p, hint=objective_hint, provider=provider)
    payload = {
        **csv_part,
        "hint": objective_hint,
        "budget": int(budget),
        "reliability_mode": reliability_mode,
        "provider": provider,
        "api_key": DEFAULT_KEY,
    }
    return _http_json("POST", "/api/run", payload)


@mcp.tool()
def get_run_status(run_id: str) -> Dict[str, Any]:
    """Fetch run status, history, diagnostics, and artifact map."""
    if not run_id or not run_id.strip():
        raise ValueError("run_id must not be empty")
    return _http_json("GET", f"/api/run/{run_id.strip()}/status")


@mcp.tool()
def get_artifact_url(run_id: str, artifact_name: str) -> Dict[str, Any]:
    """Build direct URL for a run artifact file."""
    if artifact_name not in _VALID_ARTIFACTS:
        raise ValueError(f"artifact_name must be one of {sorted(_VALID_ARTIFACTS)}, got: {artifact_name!r}")
    return {
        "artifact_name": artifact_name,
        "url": f"{API_BASE}/api/run/{run_id}/artifact/{artifact_name}",
    }


@mcp.tool()
def bootstrap_project(
    csv_path: str,
    project_name: str = "",
    user_goal: str = "",
    budget: int = 6,
    reliability_mode: str = "balanced",
    launch_run: bool = True,
    wait_for_completion: bool = False,
    poll_interval_sec: int = 5,
    max_wait_sec: int = 900,
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    """
    One-command project bootstrap:
    - health check
    - dataset profile
    - discovery recommendation (returns csv_cache_id)
    - optional run launch + optional status polling

    The csv_cache_id from discovery is threaded into the run call automatically,
    so the CSV is only transmitted once.

    Returns investor-ready summary payload with artifact URLs when available.
    """
    _check_api_key(provider)
    _validate_csv_path(csv_path)
    _validate_budget(budget)
    _validate_reliability_mode(reliability_mode)

    h = health()
    prof = profile_csv(csv_path)
    disc = discover_direction(csv_path, hint=user_goal, provider=provider)
    disc_summary = _discovery_summary(disc)
    # Reuse the cache from discover so we don't re-upload
    _cache_id = disc_summary.get("csv_cache_id") or disc.get("csv_cache_id") or ""

    objective_hint = user_goal.strip()
    if disc_summary.get("recommended_objective"):
        rec = str(disc_summary["recommended_objective"]).strip()
        objective_hint = f"{objective_hint}. {rec}".strip(". ").strip()

    result: Dict[str, Any] = {
        "project_name": project_name or Path(csv_path).stem,
        "api_base": API_BASE,
        "health": h,
        "dataset_profile": prof,
        "discovery": disc_summary,
        "proposed_objective_hint": objective_hint,
        "run": {
            "launched": False,
            "run_id": None,
            "status": None,
            "best": None,
            "diagnostics": None,
            "artifact_urls": {},
        },
        "next_steps": [
            "Review discovery objective and confirm constraints.",
            "Use reliability_mode=balanced for stable demos, aggressive for deeper search.",
            "Pull program.md, analysis.ipynb, progress.png after run completion.",
        ],
    }

    if not launch_run:
        return result

    started = start_research_run(
        csv_path=csv_path,
        objective_hint=objective_hint,
        budget=budget,
        reliability_mode=reliability_mode,
        provider=provider,
        csv_cache_id=_cache_id,
    )
    run_id = started.get("run_id")
    result["run"]["launched"] = True
    result["run"]["run_id"] = run_id
    if not run_id:
        result["run"]["error"] = started.get("error") or "run_id not returned by server"
        return result

    # Initial status fetch
    try:
        status = get_run_status(run_id)
    except Exception as e:
        result["run"]["status"] = "unknown"
        result["run"]["error"] = str(e)
        return result

    if wait_for_completion:
        deadline = time.time() + max(30, int(max_wait_sec))
        consecutive_errors = 0
        while status.get("status") not in ("done", "error") and time.time() < deadline:
            time.sleep(max(1, int(poll_interval_sec)))
            try:
                status = get_run_status(run_id)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    result["run"]["poll_error"] = f"Status polling failed 5x: {e}"
                    break

    result["run"]["status"] = status.get("status")
    result["run"]["best"] = status.get("best")
    result["run"]["diagnostics"] = status.get("diagnostics")
    result["run"]["executive_brief"] = status.get("executive_brief")
    result["run"]["artifact_urls"] = _artifact_urls(run_id, status.get("artifacts"))
    return result


@mcp.tool()
def poll_until_done(
    run_id: str,
    poll_interval_sec: int = 5,
    max_wait_sec: int = 1200,
) -> Dict[str, Any]:
    """
    Block until a run finishes (or times out), returning the final status.
    Prints progress dots to stderr so the caller knows it's alive.
    Use this after start_research_run when you need results synchronously.
    """
    import sys
    if not run_id or not run_id.strip():
        raise ValueError("run_id must not be empty")
    deadline = time.time() + max(30, int(max_wait_sec))
    status: Dict[str, Any] = {}
    consecutive_errors = 0
    last_exp = -1
    while time.time() < deadline:
        try:
            status = get_run_status(run_id)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                return {"run_id": run_id, "error": f"Status polling failed 5x: {e}", "status": "unknown"}
            time.sleep(max(1, int(poll_interval_sec)))
            continue

        run_status = status.get("status")
        # Print progress so the agent sees something is happening
        cur_exp = len(status.get("history") or [])
        if cur_exp != last_exp:
            last_exp = cur_exp
            best = (status.get("best") or {})
            metric_txt = ""
            if best.get("metric_name") and best.get("metric_val") is not None:
                metric_txt = f" | best {best['metric_name']}={best['metric_val']:.4f}"
            print(f"[poll] run={run_id} status={run_status} exp={cur_exp}{metric_txt}", file=sys.stderr, flush=True)

        if run_status in ("done", "error"):
            break
        time.sleep(max(1, int(poll_interval_sec)))

    artifact_urls = _artifact_urls(run_id, status.get("artifacts"))
    return {
        "run_id": run_id,
        "status": status.get("status"),
        "best": status.get("best"),
        "diagnostics": status.get("diagnostics"),
        "executive_brief": status.get("executive_brief"),
        "artifact_urls": artifact_urls,
        "timed_out": time.time() >= deadline and status.get("status") not in ("done", "error"),
    }


@mcp.tool()
def bootstrap_and_export_handoff(
    csv_path: str,
    project_name: str = "",
    user_goal: str = "",
    budget: int = 6,
    reliability_mode: str = "balanced",
    wait_for_completion: bool = True,
    poll_interval_sec: int = 5,
    max_wait_sec: int = 1200,
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    """
    High-level startup workflow:
    run bootstrap + generate investor-ready handoff payload.
    """
    _check_api_key(provider)
    _validate_csv_path(csv_path)
    _validate_budget(budget)
    _validate_reliability_mode(reliability_mode)
    base = bootstrap_project(
        csv_path=csv_path,
        project_name=project_name,
        user_goal=user_goal,
        budget=budget,
        reliability_mode=reliability_mode,
        launch_run=True,
        wait_for_completion=wait_for_completion,
        poll_interval_sec=poll_interval_sec,
        max_wait_sec=max_wait_sec,
        provider=provider,
    )
    run = base.get("run") or {}
    run_id = run.get("run_id")
    status: Dict[str, Any] = {}
    if run_id:
        try:
            status = get_run_status(run_id)
        except Exception:
            status = run
    artifact_urls = _artifact_urls(run_id, status.get("artifacts") if status else run.get("artifact_urls"))
    handoff_md = _handoff_markdown(
        project_name=base.get("project_name") or Path(csv_path).stem,
        csv_path=csv_path,
        status=status or run,
        run_id=run_id,
        artifact_urls=artifact_urls,
    )
    return {
        "project_name": base.get("project_name"),
        "run_id": run_id,
        "status": (status or run).get("status"),
        "best": (status or run).get("best"),
        "diagnostics": (status or run).get("diagnostics"),
        "executive_brief": (status or run).get("executive_brief"),
        "deploy_url": f"{API_BASE}/api/run/{run_id}/deploy" if run_id else "",
        "artifact_urls": artifact_urls,
        "handoff_markdown": handoff_md,
        "bootstrap": base,
    }


if __name__ == "__main__":
    mcp.run()
