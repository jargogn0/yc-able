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
        "fastmcp is required for the 19Labs MCP server. Install dependencies with: pip install -r requirements.txt"
    ) from e

API_BASE = os.environ.get("NINETEENLABS_API_BASE", "http://localhost:8019").rstrip("/")
DEFAULT_PROVIDER = os.environ.get("NINETEENLABS_PROVIDER", "claude")
DEFAULT_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

mcp = FastMCP("19labs-datascientist")


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
                return {}
            return json.loads(raw)
    except error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        msg = raw or str(e)
        raise RuntimeError(f"{method} {path} failed: {e.code} {msg}") from e
    except error.URLError as e:
        raise RuntimeError(f"Cannot reach 19Labs API at {API_BASE}: {e}") from e


def _read_csv(csv_path: str) -> str:
    p = Path(csv_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise RuntimeError(f"CSV file not found: {p}")
    return p.read_text()


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


@mcp.tool()
def health() -> Dict[str, Any]:
    """Check if local 19Labs API is reachable."""
    return _http_json("GET", "/health")


@mcp.tool()
def profile_csv(csv_path: str) -> Dict[str, Any]:
    """Lightweight local dataset profile for fast triage."""
    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"CSV not found: {p}")
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
def discover_direction(csv_path: str, hint: str = "", provider: str = DEFAULT_PROVIDER) -> Dict[str, Any]:
    """Run 19Labs discovery (interactive objective suggestions)."""
    p = Path(csv_path).expanduser().resolve()
    csv_text = _read_csv(str(p))
    payload = {
        "filename": p.name,
        "csv": csv_text,
        "hint": hint,
        "provider": provider,
        "api_key": DEFAULT_KEY,
    }
    return _http_json("POST", "/api/discover", payload)


@mcp.tool()
def start_research_run(
    csv_path: str,
    objective_hint: str = "",
    budget: int = 6,
    reliability_mode: str = "balanced",
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    """Start a full research run and return run_id."""
    p = Path(csv_path).expanduser().resolve()
    csv_text = _read_csv(str(p))
    payload = {
        "filename": p.name,
        "csv": csv_text,
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
    return _http_json("GET", f"/api/run/{run_id}/status")


@mcp.tool()
def get_artifact_url(run_id: str, artifact_name: str) -> Dict[str, Any]:
    """Build direct URL for a run artifact file."""
    allowed = {
        "program_md",
        "prepare_py",
        "analysis_ipynb",
        "progress_png",
        "results_tsv",
        "train_py",
        "final_report_md",
    }
    if artifact_name not in allowed:
        raise RuntimeError(f"Unknown artifact: {artifact_name}")
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
    poll_interval_sec: int = 3,
    max_wait_sec: int = 900,
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    """
    One-command project bootstrap:
    - health check
    - dataset profile
    - discovery recommendation
    - optional run launch + optional status polling
    Returns investor-ready summary payload with artifact URLs when available.
    """
    h = health()
    prof = profile_csv(csv_path)
    disc = discover_direction(csv_path, hint=user_goal, provider=provider)
    disc_summary = _discovery_summary(disc)
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
    )
    run_id = started.get("run_id")
    result["run"]["launched"] = True
    result["run"]["run_id"] = run_id
    if not run_id:
        return result

    status = get_run_status(run_id)
    if wait_for_completion:
        deadline = time.time() + max(30, int(max_wait_sec))
        while status.get("status") not in ("done", "error") and time.time() < deadline:
            time.sleep(max(1, int(poll_interval_sec)))
            status = get_run_status(run_id)

    result["run"]["status"] = status.get("status")
    result["run"]["best"] = status.get("best")
    result["run"]["diagnostics"] = status.get("diagnostics")
    result["run"]["executive_brief"] = status.get("executive_brief")
    result["run"]["artifact_urls"] = _artifact_urls(run_id, status.get("artifacts"))
    return result


@mcp.tool()
def bootstrap_and_export_handoff(
    csv_path: str,
    project_name: str = "",
    user_goal: str = "",
    budget: int = 6,
    reliability_mode: str = "balanced",
    wait_for_completion: bool = True,
    poll_interval_sec: int = 3,
    max_wait_sec: int = 1200,
    provider: str = DEFAULT_PROVIDER,
) -> Dict[str, Any]:
    """
    High-level startup workflow:
    run bootstrap + generate investor-ready handoff payload.
    """
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
    status = {}
    if run_id:
        status = get_run_status(run_id)
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
