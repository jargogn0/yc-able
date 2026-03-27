#!/usr/bin/env python3
"""
19Labs API Server
- POST /api/run          → start research (JSON body)
- GET  /api/run/{id}/stream → SSE stream of live logs
- GET  /api/run/{id}/status → current status + results
- GET  /api/run/{id}/deploy → download deploy.zip
- GET  /                 → serves 19labs-app.html
"""
import asyncio, base64, glob, hashlib, hmac as _hmac, json, os, secrets as _secrets, signal, shutil, sqlite3, tempfile, threading, time, uuid, zipfile
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.environ.get("NINETEENLABS_DB", Path(__file__).parent / "19labs.db"))

def _init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'running',
        filename TEXT,
        provider TEXT DEFAULT 'claude',
        started REAL,
        finished REAL,
        hint TEXT DEFAULT '',
        budget INTEGER DEFAULT 6,
        best_model TEXT,
        best_metric_name TEXT,
        best_metric_val REAL,
        total_experiments INTEGER DEFAULT 0,
        token_input INTEGER DEFAULT 0,
        token_output INTEGER DEFAULT 0,
        token_calls INTEGER DEFAULT 0,
        yc_score INTEGER,
        error TEXT,
        result_json TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        name TEXT DEFAULT '',
        password_hash TEXT NOT NULL,
        created REAL NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS user_sessions (
        token TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        created REAL NOT NULL,
        expires REAL NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS user_api_keys (
        user_id TEXT NOT NULL,
        provider TEXT NOT NULL,
        api_key TEXT NOT NULL,
        PRIMARY KEY (user_id, provider)
    )""")
    conn.commit()
    conn.close()

# ── AUTH HELPERS ──────────────────────────────────────────────
# JWT secret — derived deterministically from available env vars so it survives
# Railway container restarts/redeploys without any extra configuration.
# Priority: explicit JWT_SECRET → ANTHROPIC_API_KEY → ACCESS_PASSWORD → random (dev only)
def _derive_jwt_secret() -> str:
    for var in ("JWT_SECRET", "ANTHROPIC_API_KEY", "ACCESS_PASSWORD"):
        val = os.environ.get(var, "").strip()
        if val:
            return hashlib.sha256(f"19labs-jwt-v1:{val}".encode()).hexdigest()
    return _secrets.token_hex(32)  # fallback: only for local dev with no env vars

_JWT_SECRET = _derive_jwt_secret()

def _make_jwt(user_id: str, email: str, name: str) -> str:
    h = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
    p = base64.urlsafe_b64encode(json.dumps({"sub": user_id, "email": email, "name": name, "exp": int(time.time()) + 30*24*3600}).encode()).rstrip(b'=').decode()
    sig = base64.urlsafe_b64encode(_hmac.new(_JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()).rstrip(b'=').decode()
    return f"{h}.{p}.{sig}"

def _verify_jwt(token: str) -> dict | None:
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        h, p, s = parts
        expected = base64.urlsafe_b64encode(_hmac.new(_JWT_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()).rstrip(b'=').decode()
        if not _hmac.compare_digest(s, expected):
            return None
        data = json.loads(base64.urlsafe_b64decode(p + '=' * (4 - len(p) % 4)))
        if data.get('exp', 0) < time.time():
            return None
        return data
    except Exception:
        return None

def _hash_pw(pw: str) -> str:
    salt = _secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 260000)
    return f"{salt}:{h.hex()}"

def _verify_pw(pw: str, stored: str) -> bool:
    try:
        salt, h = stored.split(':', 1)
        return hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 260000).hex() == h
    except Exception:
        return False

def _get_user(token: str) -> dict | None:
    if not token:
        return None
    # JWT first — stateless, survives container restarts
    data = _verify_jwt(token)
    if data:
        return {"id": data["sub"], "email": data["email"], "name": data.get("name", "")}
    # Fallback: DB session (for old tokens)
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT u.id, u.email, u.name FROM users u "
            "JOIN user_sessions s ON u.id=s.user_id "
            "WHERE s.token=? AND s.expires>?",
            (token, time.time())
        ).fetchone()
        conn.close()
        return {"id": row[0], "email": row[1], "name": row[2]} if row else None
    except Exception:
        return None

def _token_from_request(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    return auth.replace("Bearer ", "").strip()

def _get_user_api_key(user_id: str, provider: str) -> str:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT api_key FROM user_api_keys WHERE user_id=? AND provider=?",
            (user_id, provider)
        ).fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    # Fallback: env vars (useful when Railway DB is fresh after redeploy)
    _env_fallbacks = {
        "claude": os.environ.get("ANTHROPIC_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "gemini": os.environ.get("GEMINI_API_KEY", ""),
    }
    return _env_fallbacks.get(provider, "")

_init_db()

def _save_run_to_db(run_id: str, run: dict):
    """Persist run summary to SQLite."""
    try:
        result = run.get("result") or {}
        best = result.get("best") or {}
        tu = result.get("token_usage") or {}
        diag = result.get("diagnostics") or {}
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""INSERT OR REPLACE INTO runs
            (id, status, filename, provider, started, finished, hint, budget,
             best_model, best_metric_name, best_metric_val, total_experiments,
             token_input, token_output, token_calls, yc_score, error, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, run.get("status", "done"),
             Path(run.get("csv", "")).name if run.get("csv") else "",
             run.get("provider", "claude"),
             run.get("started"),
             time.time(),
             run.get("hint", ""),
             run.get("budget", 6),
             best.get("model"),
             best.get("metric_name"),
             best.get("metric_val"),
             result.get("total_experiments", len(result.get("history", []))),
             tu.get("input", 0),
             tu.get("output", 0),
             tu.get("calls", 0),
             diag.get("yc_readiness_score"),
             run.get("error"),
             None))  # Don't store full result_json for now (too large)
        conn.commit()
        conn.close()
    except Exception:
        pass  # Non-critical -- don't block the run

def _load_run_history_from_db():
    """Load recent runs from SQLite for the /api/runs endpoint."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY started DESC LIMIT 100"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

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
_shared_results: dict[str, dict] = {}
# In-memory store for pre-uploaded media datasets (images / audio / zip)
_MEDIA_DATASETS: dict[str, dict] = {}
APP_HTML = Path(__file__).parent / "19labs-app.html"
LANDING_HTML = Path(__file__).parent / "landing.html"

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

# ── SERVE LANDING + APP ───────────────────────────────────────
@app.get("/")
def serve_landing():
    if LANDING_HTML.exists():
        return HTMLResponse(LANDING_HTML.read_text())
    return serve_app_page()

@app.get("/app")
def serve_app_page():
    if APP_HTML.exists():
        return HTMLResponse(APP_HTML.read_text())
    return HTMLResponse("<h1>19Labs</h1><p>Place 19labs-app.html next to server.py</p>")

# ── DIRECT DOWNLOADS ──────────────────────────────────────────
import io, zipfile, stat

_APP_URL = "https://yc-able-production.up.railway.app/app"

_INFO_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>    <string>19Labs</string>
    <key>CFBundleName</key>          <string>19Labs</string>
    <key>CFBundleDisplayName</key>   <string>19Labs</string>
    <key>CFBundleIdentifier</key>    <string>com.19labs.desktop</string>
    <key>CFBundleVersion</key>       <string>1.0.0</string>
    <key>CFBundleShortVersionString</key><string>1.0.0</string>
    <key>CFBundlePackageType</key>   <string>APPL</string>
    <key>LSMinimumSystemVersion</key><string>10.13</string>
    <key>LSUIElement</key>           <false/>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"""

_MAC_EXEC = f"""\
#!/bin/bash
# 19Labs — The Cursor for Data Science
open "{_APP_URL}"
"""

_WIN_SCRIPT = f"""\
@echo off
start "" "{_APP_URL}"
"""

_LINUX_SCRIPT = f"""\
#!/usr/bin/env bash
# 19Labs — The Cursor for Data Science
URL="{_APP_URL}"
if command -v xdg-open &>/dev/null; then xdg-open "$URL"
elif command -v open &>/dev/null; then open "$URL"
else echo "Open in browser: $URL"; fi
"""

def _build_mac_zip() -> bytes:
    """Build a real .app bundle inside a zip archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Executable — must have execute bit set
        info = zipfile.ZipInfo("19Labs.app/Contents/MacOS/19Labs")
        info.external_attr = (stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP |
                               stat.S_IROTH | stat.S_IXOTH) << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, _MAC_EXEC)
        # Info.plist
        zf.writestr("19Labs.app/Contents/Info.plist", _INFO_PLIST)
        # PkgInfo
        zf.writestr("19Labs.app/Contents/PkgInfo", "APPL????")
    buf.seek(0)
    return buf.read()

# Pre-build mac zip once at startup
_MAC_ZIP = _build_mac_zip()

_GH_RELEASE = "https://github.com/jargogn0/yc-able/releases/download/v1.0.0"
_GH_MAC   = f"{_GH_RELEASE}/19Labs-1.0.0.dmg"
_GH_WIN   = f"{_GH_RELEASE}/19Labs%20Setup%201.0.0.exe"
_GH_LINUX = f"{_GH_RELEASE}/19Labs-1.0.0.AppImage"

@app.get("/download")
async def download_auto(request: Request):
    ua = request.headers.get("user-agent", "").lower()
    if "windows" in ua or "win64" in ua or "win32" in ua:
        return RedirectResponse(_GH_WIN, status_code=302)
    if "mac" in ua or "darwin" in ua:
        return RedirectResponse(_GH_MAC, status_code=302)
    return RedirectResponse(_GH_LINUX, status_code=302)

@app.get("/download/mac")
def _download_mac():
    return RedirectResponse(_GH_MAC, status_code=302)

@app.get("/download/win")
def _download_windows():
    return RedirectResponse(_GH_WIN, status_code=302)

@app.get("/download/linux")
def _download_linux():
    return RedirectResponse(_GH_LINUX, status_code=302)

# ── AUTH ENDPOINTS ────────────────────────────────────────────

@app.post("/auth/signup")
async def auth_signup(request: Request):
    data = await request.json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name = data.get("name", "").strip()
    if not email or "@" not in email:
        raise HTTPException(400, "Invalid email")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    user_id = str(uuid.uuid4())
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            "INSERT INTO users (id, email, name, password_hash, created) VALUES (?,?,?,?,?)",
            (user_id, email, name, _hash_pw(password), time.time())
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(409, "Email already registered")
    conn.close()
    token = _make_jwt(user_id, email, name)
    return {"token": token, "user": {"id": user_id, "email": email, "name": name}}

@app.post("/auth/login")
async def auth_login(request: Request):
    data = await request.json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute(
        "SELECT id, email, name, password_hash FROM users WHERE email=?", (email,)
    ).fetchone()
    conn.close()
    if not row or not _verify_pw(password, row[3]):
        raise HTTPException(401, "Invalid email or password")
    token = _make_jwt(row[0], row[1], row[2])
    return {"token": token, "user": {"id": row[0], "email": row[1], "name": row[2]}}

@app.get("/auth/me")
async def auth_me(request: Request):
    user = _get_user(_token_from_request(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    return user

@app.post("/auth/logout")
async def auth_logout(request: Request):
    token = _token_from_request(request)
    if token:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("DELETE FROM user_sessions WHERE token=?", (token,))
        conn.commit()
        conn.close()
    return {"ok": True}

@app.get("/auth/keys")
async def auth_get_keys(request: Request):
    user = _get_user(_token_from_request(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT provider, api_key FROM user_api_keys WHERE user_id=?", (user["id"],)
    ).fetchall()
    conn.close()
    result = {}
    for provider, key in rows:
        result[provider] = (key[:8] + "…" + key[-4:]) if len(key) > 12 else "****"
    return result

@app.post("/auth/keys")
async def auth_save_keys(request: Request):
    user = _get_user(_token_from_request(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    data = await request.json()
    conn = sqlite3.connect(str(DB_PATH))
    for provider, key in data.items():
        # normalize: 'anthropic' → 'claude' to match engine provider names
        if provider == "anthropic":
            provider = "claude"
        if provider in ("claude", "openai", "gemini") and key and key.strip():
            conn.execute(
                "INSERT OR REPLACE INTO user_api_keys (user_id, provider, api_key) VALUES (?,?,?)",
                (user["id"], provider, key.strip())
            )
    conn.commit()
    conn.close()
    return {"ok": True}

# ── START RUN (JSON body) ──────────────────────────────────────
class RunRequest(BaseModel):
    filename: str
    csv: str = ""          # full CSV text (empty when dataset_id is set)
    dataset_id: str = ""   # pre-uploaded media dataset (images / audio)
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
    csv: str = ""
    dataset_id: str = ""   # pre-uploaded media dataset
    hint: str = ""
    previous_objective: dict = {}  # what the agent was proposing before the user corrected it
    api_key: str = ""
    provider: str = "claude"

class ChatRequest(BaseModel):
    message: str
    api_key: str = ""
    provider: str = "claude"
    context: dict = {}

class PredictRequest(BaseModel):
    data: list[dict] = []  # list of rows as dicts
    csv_text: str = ""     # alternative: raw CSV text

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

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma"}

def _profile_media_dir(data_dir: Path) -> dict:
    """Scan a directory of images/audio and return dataset info."""
    data_dir = Path(data_dir)
    all_files = [f for f in data_dir.rglob("*") if f.is_file() and not f.name.startswith(".")]
    image_files = [f for f in all_files if f.suffix.lower() in _IMAGE_EXTS]
    audio_files = [f for f in all_files if f.suffix.lower() in _AUDIO_EXTS]
    media_files = image_files if image_files else audio_files
    media_type = "image" if image_files else ("audio" if audio_files else "unknown")

    classes: dict[str, int] = {}
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            cls_files = [f for f in item.rglob("*") if f.is_file() and f.suffix.lower() in (_IMAGE_EXTS if image_files else _AUDIO_EXTS)]
            if cls_files:
                classes[item.name] = len(cls_files)

    task_type = f"{media_type}_classification" if classes else media_type
    return {
        "type": task_type,
        "media_type": media_type,
        "data_dir": str(data_dir),
        "total_files": len(media_files),
        "classes": classes,
        "num_classes": len(classes),
        "sample_files": [str(f.relative_to(data_dir)) for f in media_files[:6]],
    }


@app.post("/api/upload-dataset")
async def upload_media_dataset(file: UploadFile = File(...)):
    """Accept ZIP/image/audio dataset. Returns dataset_id for use in /api/run."""
    dataset_id = str(uuid.uuid4())[:10]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"19labs_media_{dataset_id}_"))
    filename = file.filename or "dataset.zip"
    ext = Path(filename).suffix.lower()
    content = await file.read()

    if ext == ".zip":
        zip_path = tmp_dir / filename
        zip_path.write_bytes(content)
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        zip_path.unlink()
        # Unwrap single top-level folder if that's all there is
        items = [x for x in data_dir.iterdir()]
        if len(items) == 1 and items[0].is_dir():
            data_dir = items[0]
    else:
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        (data_dir / filename).write_bytes(content)

    info = _profile_media_dir(data_dir)
    _MEDIA_DATASETS[dataset_id] = {"path": str(data_dir), "filename": filename, **info}
    return {"dataset_id": dataset_id, **info}


@app.post("/api/run")
async def start_run(req: RunRequest, request: Request):
    # Auto-fill API key from user account if not provided
    if not req.api_key:
        user = _get_user(_token_from_request(request))
        if user:
            saved = _get_user_api_key(user["id"], req.provider)
            if saved:
                req.api_key = saved

    run_id = str(uuid.uuid4())[:12]
    ws = Path(tempfile.mkdtemp(prefix=f"19labs_{run_id}_"))

    if req.dataset_id:
        # Media dataset (images / audio) — use pre-uploaded directory
        media = _MEDIA_DATASETS.get(req.dataset_id)
        if not media:
            raise HTTPException(404, "Media dataset not found. Please re-upload.")
        csv_path = Path(media["path"])  # directory, not a CSV file
    else:
        if len(req.csv.encode("utf-8")) > CSV_MAX_BYTES:
            raise HTTPException(413, f"CSV too large. Max 10 MB, got {len(req.csv.encode('utf-8')) / (1024*1024):.1f} MB.")
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
                _save_run_to_db(run_id, RUNS[run_id])
            elif RUNS[run_id]["status"] != "error":
                RUNS[run_id]["status"] = "done"
                _save_run_to_db(run_id, RUNS[run_id])
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
            _save_run_to_db(run_id, RUNS[run_id])

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
    ws = Path(tempfile.mkdtemp(prefix="19labs_discover_"))
    cleanup_ws = True
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from engine import discover_user_need, profile_dataset, profile_media_dataset

        # Resolve data path — media dataset or CSV
        if req.dataset_id:
            media = _MEDIA_DATASETS.get(req.dataset_id)
            if not media:
                return {"ok": False, "error": "Media dataset not found. Please re-upload."}
            data_path = media["path"]
            profile = profile_media_dataset(data_path)
            cleanup_ws = False  # don't delete the media dataset dir
        else:
            if len(req.csv.encode("utf-8")) > CSV_MAX_BYTES:
                raise HTTPException(413, f"CSV too large. Max 10 MB, got {len(req.csv.encode('utf-8')) / (1024*1024):.1f} MB.")
            csv_path = ws / req.filename
            csv_path.write_text(req.csv)
            data_path = str(csv_path)
            profile = profile_dataset(data_path)

        resolved_api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        if not resolved_api_key:
            fallback = _fallback_discovery(profile, req.hint)
            fallback["ok"] = True
            fallback["provider_note"] = "Using smart fallback discovery. Add an API key for richer AI analysis."
            return fallback

        result = discover_user_need(data_path, user_hint=req.hint, previous_objective=req.previous_objective, api_key=resolved_api_key, provider=req.provider or "claude")
        result["used_fallback"] = False
        return {"ok": True, **result}
    except Exception as e:
        msg = str(e)
        if msg.strip() == "Connection error.":
            msg = "API connection error during discovery. Check internet/VPN/firewall/proxy."
        return {"ok": False, "error": msg, "used_fallback": False}
    finally:
        if cleanup_ws:
            shutil.rmtree(ws, ignore_errors=True)

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
    # In-memory runs (current session)
    for rid, run in sorted(RUNS.items(), key=lambda x: x[1].get("started", 0), reverse=True):
        out.append({
            "id": rid,
            "status": run["status"],
            "started": run.get("started"),
            "filename": Path(run.get("csv", "")).name if run.get("csv") else "",
            "provider": run.get("provider", "claude"),
            "best_model": (run.get("result") or {}).get("best", {}).get("model"),
            "best_metric": (run.get("result") or {}).get("best", {}).get("metric_val"),
        })
    # Historical runs from DB (not in memory)
    in_memory_ids = set(RUNS.keys())
    for row in _load_run_history_from_db():
        if row["id"] not in in_memory_ids:
            out.append({
                "id": row["id"],
                "status": row["status"],
                "started": row["started"],
                "filename": row.get("filename", ""),
                "provider": row.get("provider", "claude"),
                "best_model": row.get("best_model"),
                "best_metric": row.get("best_metric_val"),
                "historical": True,  # flag that workspace is gone
            })
    out.sort(key=lambda x: x.get("started") or 0, reverse=True)
    return {"runs": out[:100]}

@app.get("/api/stats")
def get_stats():
    """Dashboard stats -- total runs, models trained, best scores."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        successful = conn.execute("SELECT COUNT(*) FROM runs WHERE status='done' AND best_model IS NOT NULL").fetchone()[0]
        total_experiments = conn.execute("SELECT COALESCE(SUM(total_experiments), 0) FROM runs").fetchone()[0]
        total_tokens = conn.execute("SELECT COALESCE(SUM(token_input + token_output), 0) FROM runs").fetchone()[0]
        conn.close()
        return {
            "total_runs": total + len(RUNS),
            "successful_runs": successful,
            "total_experiments": int(total_experiments),
            "total_tokens": int(total_tokens),
        }
    except Exception:
        return {"total_runs": len(RUNS), "successful_runs": 0, "total_experiments": 0, "total_tokens": 0}

# ── SSE STREAM ─────────────────────────────────────────────────
@app.get("/api/run/{run_id}/stream")
async def stream_logs(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")

    async def generate():
        sent = 0
        run_start = RUNS[run_id].get("started", time.time())
        last_data_ts = time.time()
        KEEPALIVE_INTERVAL = 15  # send ping every 15s of silence to prevent proxy timeout

        while True:
            run = RUNS[run_id]
            logs = run["logs"]
            flushed = 0
            while sent < len(logs):
                entry = dict(logs[sent])
                entry["elapsed"] = round(time.time() - run_start, 1)
                yield f"data: {json.dumps(entry)}\n\n"
                sent += 1
                flushed += 1
            if flushed:
                last_data_ts = time.time()
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
                        last_data_ts = time.time()
                    if run["status"] in ("done", "error"):
                        break
                yield f"data: {json.dumps({'tag':'sys','msg':'__DONE__','ts':time.strftime('%H:%M:%S'),'elapsed':round(time.time()-run_start,1)})}\n\n"
                break
            # Keepalive ping — prevents Railway/nginx from closing idle SSE connection
            if time.time() - last_data_ts >= KEEPALIVE_INTERVAL:
                yield f": keepalive\n\n"
                last_data_ts = time.time()
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

# ── INLINE CHART GENERATION ────────────────────────────────────
class ChartRequest(BaseModel):
    query: str           # e.g. "show me revenue by month"
    csv_text: str = ""   # raw CSV data
    headers: list[str] = []
    sample_rows: list[list[str]] = []
    api_key: str = ""
    provider: str = "claude"

@app.post("/api/chart")
async def generate_chart(req: ChartRequest):
    """Generate a chart from natural language + data. Returns chart spec for client-side rendering."""
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(400, "API key required")

    import pandas as pd, io, numpy as np

    # Build a dataframe from whatever we have
    if req.csv_text:
        df = pd.read_csv(io.StringIO(req.csv_text), nrows=5000)
    elif req.headers and req.sample_rows:
        df = pd.DataFrame(req.sample_rows, columns=req.headers)
    else:
        raise HTTPException(400, "Provide csv_text or headers+sample_rows")

    # Build a rich data summary for the LLM
    col_info = []
    for col in df.columns:
        s = df[col].dropna()
        info = {"name": col, "dtype": str(df[col].dtype)}
        if pd.api.types.is_numeric_dtype(s):
            info.update({"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()), "type": "numeric"})
        else:
            info.update({"unique": int(s.nunique()), "top3": s.value_counts().head(3).to_dict(), "type": "categorical"})
        col_info.append(info)

    from engine import ask, _init_client
    _init_client(api_key, req.provider or "claude")

    prompt = f"""Given this dataset and query, generate a chart specification.

QUERY: {req.query}

COLUMNS ({len(df.columns)}):
{json.dumps(col_info, indent=1, default=str)[:3000]}

SAMPLE (first 3 rows):
{df.head(3).to_string(index=False)[:1000]}

Return STRICT JSON with this format:
{{
  "chart_type": "bar|line|scatter|histogram|pie|heatmap|box",
  "title": "Chart Title",
  "x": {{"column": "col_name", "label": "X Label"}},
  "y": {{"column": "col_name", "label": "Y Label", "agg": "sum|mean|count|max|min|none"}},
  "color": {{"column": "col_name_or_null"}},
  "insight": "One sentence insight about the data",
  "sort": "asc|desc|none",
  "limit": 20
}}

Pick the chart type that best answers the query. If the user asks about distribution, use histogram. If trends over time, use line. If comparison, use bar. If relationship, use scatter.
For aggregations: if y needs grouping by x, set agg (sum/mean/count). If raw values, set "none".
Return ONLY the JSON object, no markdown."""

    try:
        raw = ask("You generate chart specifications from natural language. Return only valid JSON.", prompt, 800)
        # Parse JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            return {"ok": False, "error": "Could not parse chart spec"}
        spec = json.loads(json_match.group())

        # Now compute the actual data for the chart
        chart_data = _compute_chart_data(df, spec)
        spec["data"] = chart_data
        spec["total_rows"] = len(df)
        return {"ok": True, "spec": spec}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _compute_chart_data(df, spec):
    """Compute the actual chart data arrays from the dataframe and spec."""
    import pandas as pd, numpy as np

    x_col = spec.get("x", {}).get("column", "")
    y_col = spec.get("y", {}).get("column", "")
    agg = spec.get("y", {}).get("agg", "none")
    color_col = spec.get("color", {}).get("column")
    chart_type = spec.get("chart_type", "bar")
    limit = min(spec.get("limit", 20), 50)
    sort = spec.get("sort", "none")

    try:
        if chart_type == "histogram":
            col = y_col or x_col
            if col and col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                counts, edges = np.histogram(s, bins=min(30, max(10, len(s) // 20)))
                labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(counts))]
                return {"labels": labels, "values": counts.tolist()}

        if chart_type == "pie":
            col = x_col or y_col
            if col and col in df.columns:
                vc = df[col].value_counts().head(limit)
                return {"labels": vc.index.tolist(), "values": vc.values.tolist()}

        if x_col and x_col in df.columns:
            if agg and agg != "none" and y_col and y_col in df.columns:
                grouped = df.groupby(x_col, dropna=True)[y_col].agg(agg).reset_index()
                if sort == "desc":
                    grouped = grouped.sort_values(y_col, ascending=False)
                elif sort == "asc":
                    grouped = grouped.sort_values(y_col, ascending=True)
                grouped = grouped.head(limit)
                labels = grouped[x_col].astype(str).tolist()
                values = grouped[y_col].tolist()
            elif y_col and y_col in df.columns:
                sub = df[[x_col, y_col]].dropna().head(limit * 10)
                labels = sub[x_col].astype(str).tolist()
                values = pd.to_numeric(sub[y_col], errors="coerce").tolist()
            else:
                vc = df[x_col].value_counts().head(limit)
                labels = vc.index.astype(str).tolist()
                values = vc.values.tolist()

            # Convert any numpy types
            values = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in values]
            return {"labels": labels, "values": values}

    except Exception:
        pass

    return {"labels": [], "values": []}

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

# ── BATCH PREDICTION ───────────────────────────────────────────
@app.post("/api/run/{run_id}/predict")
async def predict(run_id: str, req: PredictRequest):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    run = RUNS[run_id]
    result = run.get("result")
    if not result:
        raise HTTPException(400, "Run not finished yet")
    ws = Path(run.get("ws", ""))
    deploy_path = result.get("deploy_path")
    dp = Path(deploy_path) if deploy_path and Path(deploy_path).exists() else ws

    # Find model file
    model_path = None
    for pattern in ["model.pkl", "*.pkl", "*.joblib"]:
        for f in list(dp.glob(pattern)) + list(ws.glob(pattern)):
            if f.is_file():
                model_path = f
                break
        if model_path:
            break

    if not model_path:
        raise HTTPException(404, "No trained model found. Run may not have completed successfully.")

    import joblib
    import pandas as pd
    import numpy as np

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

    # Build dataframe from input
    if req.csv_text:
        import io
        df = pd.read_csv(io.StringIO(req.csv_text))
    elif req.data:
        df = pd.DataFrame(req.data)
    else:
        raise HTTPException(400, "Provide either 'data' (list of dicts) or 'csv_text'")

    obj = result.get("objective") or {}
    target = obj.get("target", "")

    # Drop target column if present (user might include it)
    if target and target in df.columns:
        df = df.drop(columns=[target])

    try:
        # Try direct prediction first
        predictions = model.predict(df)
        # Convert numpy to python types
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()

        # If classification, also get probabilities
        proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba_raw = model.predict_proba(df)
                if hasattr(proba_raw, 'tolist'):
                    proba = proba_raw.tolist()
            except Exception:
                pass

        response = {
            "ok": True,
            "predictions": predictions,
            "count": len(predictions),
            "model": result.get("best", {}).get("model", "Unknown"),
            "target": target,
        }
        if proba is not None:
            response["probabilities"] = proba
        return response

    except Exception as e:
        # If direct prediction fails, try with basic preprocessing
        error_msg = str(e)
        # Common issue: categorical columns need encoding
        try:
            # Try label encoding categoricals
            from sklearn.preprocessing import LabelEncoder
            df_encoded = df.copy()
            for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            df_encoded = df_encoded.fillna(0)
            predictions = model.predict(df_encoded)
            if hasattr(predictions, 'tolist'):
                predictions = predictions.tolist()
            return {
                "ok": True,
                "predictions": predictions,
                "count": len(predictions),
                "model": result.get("best", {}).get("model", "Unknown"),
                "target": target,
                "note": "Applied automatic preprocessing (label encoding + null fill)"
            }
        except Exception as e2:
            raise HTTPException(400, f"Prediction failed: {error_msg}. Auto-fix also failed: {e2}")


@app.post("/api/run/{run_id}/predict-csv")
async def predict_csv(run_id: str, request: Request):
    body = await request.body()
    csv_text = body.decode("utf-8")
    req = PredictRequest(csv_text=csv_text)
    return await predict(run_id, req)


# ── TIER 2: URL Data Connector ────────────────────────────────

class FetchURLRequest(BaseModel):
    url: str

@app.post("/api/fetch-url")
async def fetch_url_data(req: FetchURLRequest):
    """Fetch CSV/JSON data from a URL."""
    import urllib.request, io
    try:
        url = req.url.strip()
        # Google Sheets → export as CSV
        if 'docs.google.com/spreadsheets' in url:
            # Extract sheet ID and convert to CSV export URL
            import re
            match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
            if match:
                sheet_id = match.group(1)
                url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

        req_obj = urllib.request.Request(url, headers={"User-Agent": "19Labs/1.0"})
        with urllib.request.urlopen(req_obj, timeout=30) as resp:
            data = resp.read()
            if len(data) > 10 * 1024 * 1024:
                raise HTTPException(413, "File too large (>10MB)")
            text = data.decode("utf-8", errors="replace")

        # Detect if JSON
        text_stripped = text.strip()
        if text_stripped.startswith('[') or text_stripped.startswith('{'):
            import pandas as pd
            df = pd.read_json(io.StringIO(text_stripped))
            text = df.to_csv(index=False)
            return {"ok": True, "csv": text, "filename": "fetched_data.csv", "converted_from": "json", "rows": len(df)}

        return {"ok": True, "csv": text, "filename": "fetched_data.csv"}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── TIER 2: Data Transform ───────────────────────────────────

class TransformRequest(BaseModel):
    csv_text: str
    transforms: list[dict]  # [{"column": "col1", "action": "drop_nulls"}, ...]

@app.post("/api/transform")
async def transform_data(req: TransformRequest):
    """Apply column-level transforms to data and return modified CSV."""
    import pandas as pd, io, numpy as np
    try:
        df = pd.read_csv(io.StringIO(req.csv_text))
        log = []
        for t in req.transforms:
            col = t.get("column", "")
            action = t.get("action", "")
            if col and col not in df.columns:
                log.append(f"Column '{col}' not found, skipped")
                continue

            if action == "drop_nulls":
                before = len(df)
                df = df.dropna(subset=[col])
                log.append(f"Dropped {before - len(df)} rows with null '{col}'")
            elif action == "fill_mean":
                mean_val = pd.to_numeric(df[col], errors="coerce").mean()
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(mean_val)
                log.append(f"Filled nulls in '{col}' with mean ({mean_val:.4f})")
            elif action == "fill_median":
                median_val = pd.to_numeric(df[col], errors="coerce").median()
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median_val)
                log.append(f"Filled nulls in '{col}' with median ({median_val:.4f})")
            elif action == "fill_mode":
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else ""
                df[col] = df[col].fillna(mode_val)
                log.append(f"Filled nulls in '{col}' with mode ({mode_val})")
            elif action == "one_hot":
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                log.append(f"One-hot encoded '{col}' → {len(dummies.columns)} new columns")
            elif action == "label_encode":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                log.append(f"Label encoded '{col}' ({len(le.classes_)} classes)")
            elif action == "log_transform":
                df[col] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0))
                log.append(f"Applied log1p transform to '{col}'")
            elif action == "standardize":
                s = pd.to_numeric(df[col], errors="coerce")
                df[col] = (s - s.mean()) / (s.std() + 1e-10)
                log.append(f"Standardized '{col}' (mean=0, std=1)")
            elif action == "bin":
                bins = t.get("bins", 5)
                df[col + "_binned"] = pd.qcut(pd.to_numeric(df[col], errors="coerce"), q=bins, labels=False, duplicates="drop")
                log.append(f"Binned '{col}' into {bins} quantile groups")
            elif action == "drop_column":
                df = df.drop(columns=[col])
                log.append(f"Dropped column '{col}'")
            elif action == "to_numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                log.append(f"Converted '{col}' to numeric")
            elif action == "to_datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
                # Extract useful features
                df[col + "_year"] = df[col].dt.year
                df[col + "_month"] = df[col].dt.month
                df[col + "_dayofweek"] = df[col].dt.dayofweek
                df = df.drop(columns=[col])
                log.append(f"Extracted date features from '{col}' (year, month, dayofweek)")

        csv_out = df.to_csv(index=False)
        return {"ok": True, "csv": csv_out, "log": log, "rows": len(df), "cols": len(df.columns)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── TIER 2: SHAP Explainability ──────────────────────────────

@app.get("/api/run/{run_id}/explain")
async def explain_model(run_id: str):
    """Generate SHAP-based model explainability data."""
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    run = RUNS[run_id]
    result = run.get("result")
    if not result:
        raise HTTPException(400, "Run not finished yet")

    ws = Path(run.get("ws", ""))
    deploy_path = result.get("deploy_path")
    dp = Path(deploy_path) if deploy_path and Path(deploy_path).exists() else ws

    # Find model
    model_path = None
    for pattern in ["model.pkl", "*.pkl", "*.joblib"]:
        for f in list(dp.glob(pattern)) + list(ws.glob(pattern)):
            if f.is_file():
                model_path = f
                break
        if model_path:
            break

    if not model_path:
        return {"ok": False, "error": "No model found"}

    import joblib, pandas as pd, numpy as np
    try:
        model = joblib.load(model_path)

        # Load training data for SHAP background
        csv_path = run.get("csv", "")
        if not csv_path or not Path(csv_path).exists():
            return {"ok": False, "error": "Training data not found"}

        df = pd.read_csv(csv_path, nrows=500)
        obj = result.get("objective") or {}
        target = obj.get("target", "")
        if target and target in df.columns:
            df = df.drop(columns=[target])

        # Encode categoricals
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            df[col] = pd.factorize(df[col])[0]
        df = df.fillna(0)

        # Try to compute feature importance
        feature_names = list(df.columns)
        importance = {}

        # Method 1: Built-in feature_importances_
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            if len(imp) == len(feature_names):
                importance = {feature_names[i]: float(imp[i]) for i in range(len(imp))}

        # Method 2: coef_ (linear models)
        elif hasattr(model, "coef_"):
            coef = np.abs(model.coef_).flatten()
            if len(coef) == len(feature_names):
                importance = {feature_names[i]: float(coef[i]) for i in range(len(coef))}

        # Method 3: Try SHAP
        shap_values_data = None
        try:
            import shap
            sample = df.head(100)
            if hasattr(model, "predict"):
                explainer = shap.Explainer(model, sample, feature_names=feature_names)
                shap_vals = explainer(sample)
                # Mean absolute SHAP values per feature
                mean_shap = np.abs(shap_vals.values).mean(axis=0)
                if len(mean_shap.shape) > 1:
                    mean_shap = mean_shap.mean(axis=1)
                importance = {feature_names[i]: float(mean_shap[i]) for i in range(min(len(feature_names), len(mean_shap)))}
                # Top feature interactions
                shap_values_data = {
                    "sample_size": len(sample),
                    "method": "shap"
                }
        except Exception:
            pass  # SHAP not available or failed

        if not importance:
            return {"ok": False, "error": "Could not compute feature importance for this model type"}

        # Sort by importance
        sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_imp[:20]

        # Compute basic partial dependence for top 3 features
        pdp_data = {}
        try:
            for feat_name, _ in top_features[:3]:
                if feat_name not in df.columns:
                    continue
                col_vals = df[feat_name].dropna()
                if len(col_vals) < 10:
                    continue
                grid = np.linspace(col_vals.quantile(0.05), col_vals.quantile(0.95), 20)
                predictions = []
                for val in grid:
                    temp_df = df.head(50).copy()
                    temp_df[feat_name] = val
                    try:
                        preds = model.predict(temp_df)
                        predictions.append(float(np.mean(preds)))
                    except Exception:
                        break
                if len(predictions) == len(grid):
                    pdp_data[feat_name] = {
                        "grid": grid.tolist(),
                        "predictions": predictions
                    }
        except Exception:
            pass

        return {
            "ok": True,
            "feature_importance": [{"feature": k, "importance": v} for k, v in top_features],
            "partial_dependence": pdp_data,
            "method": shap_values_data.get("method", "builtin") if shap_values_data else "builtin",
            "model_type": type(model).__name__,
            "n_features": len(feature_names),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── TIER 2: Share Results ─────────────────────────────────────

@app.post("/api/run/{run_id}/share")
async def share_results(run_id: str):
    """Generate a shareable snapshot of run results."""
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    run = RUNS[run_id]
    result = run.get("result")
    if not result:
        raise HTTPException(400, "Run not finished yet")

    share_id = str(uuid.uuid4())[:8]
    best = result.get("best") or {}
    obj = result.get("objective") or {}
    diagnostics = result.get("diagnostics") or {}
    history = result.get("history") or []

    _shared_results[share_id] = {
        "share_id": share_id,
        "created": time.time(),
        "run_id": run_id,
        "filename": Path(run.get("csv", "")).name if run.get("csv") else "",
        "best": best,
        "objective": obj,
        "diagnostics": diagnostics,
        "history": [{"num": h.get("num"), "model": h.get("model"), "metric_name": h.get("metric_name"), "metric_val": h.get("metric_val"), "success": h.get("success"), "is_best": h.get("is_best")} for h in history],
        "report": result.get("report", ""),
        "executive_brief": result.get("executive_brief") or diagnostics.get("executive_brief", ""),
        "token_usage": result.get("token_usage", {}),
    }

    return {"ok": True, "share_id": share_id, "url": f"/shared/{share_id}"}


@app.get("/shared/{share_id}")
async def view_shared(share_id: str):
    """Render a shared results page."""
    if share_id not in _shared_results:
        return HTMLResponse("<h1>Not found</h1><p>This shared result has expired or does not exist.</p>", status_code=404)

    data = _shared_results[share_id]
    best = data.get("best", {})
    obj = data.get("objective", {})
    diag = data.get("diagnostics", {})
    history = data.get("history", [])

    # Build a standalone HTML page
    experiments_html = ""
    for h in history:
        if h.get("success"):
            cls = "best" if h.get("is_best") else ""
            experiments_html += f'<div class="exp {cls}"><span class="num">#{h["num"]}</span><span class="model">{h.get("model","?")}</span><span class="val">{h.get("metric_name","metric")}={float(h.get("metric_val",0)):.4f}</span></div>'

    score = diag.get("yc_readiness_score", "N/A")
    grade = diag.get("yc_grade", "N/A")
    brief = data.get("executive_brief", "")
    report = data.get("report", "")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>19Labs Results — {best.get('model','ML Model')}</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Inter',system-ui,sans-serif;background:#0a0a0b;color:#fafafa;padding:40px 20px;max-width:720px;margin:0 auto}}
h1{{font-size:24px;font-weight:800;margin-bottom:4px}}h2{{font-size:16px;font-weight:700;margin:24px 0 8px;color:#a1a1aa}}
.sub{{color:#71717a;font-size:14px;margin-bottom:24px}}.metric{{font-size:32px;font-weight:900;color:#3b82f6;margin:8px 0}}
.card{{background:#111113;border:1px solid #27272a;border-radius:12px;padding:16px 20px;margin-bottom:12px}}
.row{{display:flex;justify-content:space-between;padding:6px 0;font-size:13px;border-bottom:1px solid #1e1e22}}.row:last-child{{border:none}}
.label{{color:#71717a}}.value{{font-weight:600;font-family:'JetBrains Mono',monospace}}
.exp{{display:flex;gap:12px;align-items:center;padding:8px 12px;border-radius:8px;margin-bottom:4px;font-size:13px;background:#111113;border:1px solid #27272a}}
.exp.best{{border-color:rgba(59,130,246,.3);background:rgba(59,130,246,.05)}}
.num{{font-family:'JetBrains Mono',monospace;color:#71717a;font-size:11px;width:30px}}.model{{flex:1;font-weight:600}}.val{{font-family:'JetBrains Mono',monospace;color:#a1a1aa;font-size:12px}}
.report{{font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.8;color:#a1a1aa;white-space:pre-wrap;background:#111113;border:1px solid #27272a;border-radius:12px;padding:16px;max-height:400px;overflow-y:auto}}
.badge{{display:inline-block;padding:4px 12px;border-radius:999px;font-size:11px;font-weight:700;background:rgba(59,130,246,.1);color:#3b82f6;border:1px solid rgba(59,130,246,.3)}}
.footer{{margin-top:40px;text-align:center;color:#3f3f46;font-size:12px}}
.footer a{{color:#3b82f6;text-decoration:none}}
</style></head><body>
<h1>{best.get('model', 'ML Model')}</h1>
<div class="sub">Generated by 19Labs Autonomous ML Research · {data.get('filename','')}</div>
<div class="card">
<div class="metric">{best.get('metric_name','metric').upper()} = {float(best.get('metric_val',0)):.4f}</div>
<div class="row"><span class="label">Task</span><span class="value">{obj.get('task','N/A')}</span></div>
<div class="row"><span class="label">Target</span><span class="value">{obj.get('target','N/A')}</span></div>
<div class="row"><span class="label">Quality Score</span><span class="value">{score}/100 ({grade})</span></div>
<div class="row"><span class="label">Experiments</span><span class="value">{len(history)}</span></div>
</div>
{f'<h2>Executive Brief</h2><div class="card" style="color:#a1a1aa;font-size:13px;line-height:1.7">{brief}</div>' if brief else ''}
<h2>Experiments</h2>
{experiments_html}
{f'<h2>Report</h2><div class="report">{report[:3000]}</div>' if report else ''}
<div class="footer">Powered by <a href="/">19Labs</a> · Autonomous ML Research</div>
</body></html>"""

    return HTMLResponse(html)


# ── TIER 2: Multi-dataset Join ────────────────────────────────

class JoinRequest(BaseModel):
    datasets: list[dict]  # [{"name": "file1.csv", "csv": "..."}, ...]
    hint: str = ""
    api_key: str = ""
    provider: str = "claude"

@app.post("/api/join-datasets")
async def join_datasets(req: JoinRequest):
    """AI-powered multi-dataset join. Detects join keys automatically."""
    import pandas as pd, io
    if len(req.datasets) < 2:
        return {"ok": False, "error": "Need at least 2 datasets to join"}

    dfs = {}
    for ds in req.datasets:
        try:
            dfs[ds["name"]] = pd.read_csv(io.StringIO(ds["csv"]))
        except Exception as e:
            return {"ok": False, "error": f"Failed to parse {ds['name']}: {e}"}

    # Try to find common columns for joining
    names = list(dfs.keys())
    all_cols = {name: set(df.columns) for name, df in dfs.items()}

    # Find pairwise common columns
    common = set.intersection(*all_cols.values()) if len(all_cols) > 1 else set()

    # Smart join key detection
    join_key = None
    join_candidates = []
    for col in common:
        # Check if it looks like an ID/key column
        is_key = any([
            col.lower().endswith('_id'),
            col.lower().endswith('id'),
            col.lower() in ('id', 'key', 'index', 'code', 'name', 'date', 'timestamp'),
            all(dfs[n][col].nunique() > len(dfs[n]) * 0.5 for n in names),  # High cardinality
        ])
        if is_key:
            join_candidates.append(col)

    if not join_candidates and common:
        join_candidates = list(common)[:3]

    if not join_candidates:
        # No common columns — suggest concat
        try:
            result = pd.concat(list(dfs.values()), ignore_index=True)
            csv_out = result.to_csv(index=False)
            return {
                "ok": True,
                "csv": csv_out,
                "method": "concat",
                "join_key": None,
                "rows": len(result),
                "cols": len(result.columns),
                "log": f"No common columns found. Concatenated {len(dfs)} datasets vertically ({len(result)} rows)."
            }
        except Exception as e:
            return {"ok": False, "error": f"Join failed: {e}"}

    join_key = join_candidates[0]

    # Perform sequential left join
    result = list(dfs.values())[0]
    for i, (name, df) in enumerate(list(dfs.items())[1:]):
        # Rename overlapping columns (except join key)
        overlap = set(result.columns) & set(df.columns) - {join_key}
        if overlap:
            suffix = f"_{name.split('.')[0]}"
            df = df.rename(columns={c: c + suffix for c in overlap})
        result = result.merge(df, on=join_key, how="outer")

    csv_out = result.to_csv(index=False)
    return {
        "ok": True,
        "csv": csv_out,
        "method": "merge",
        "join_key": join_key,
        "join_candidates": join_candidates,
        "rows": len(result),
        "cols": len(result.columns),
        "log": f"Joined {len(dfs)} datasets on '{join_key}' → {len(result)} rows, {len(result.columns)} columns."
    }


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
