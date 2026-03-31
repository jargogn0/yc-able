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
import urllib.request, urllib.parse, urllib.error
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# DB path: explicit env var → Railway /data volume (persistent) → local fallback
def _resolve_db_path() -> Path:
    if os.environ.get("NINETEENLABS_DB"):
        return Path(os.environ["NINETEENLABS_DB"])
    # Railway mounts persistent volumes at /data — use it automatically if present
    railway_data = Path("/data")
    if railway_data.exists() and railway_data.is_dir():
        return railway_data / "19labs.db"
    return Path(__file__).parent / "19labs.db"

DB_PATH = _resolve_db_path()
DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists before SQLite opens it
_pg_mode = bool(os.environ.get("DATABASE_URL", "").strip())
print(f"[db] Backend: {'PostgreSQL (DATABASE_URL)' if _pg_mode else f'SQLite at {DB_PATH}'}", flush=True)

# Models dir: use same /data persistent volume when available (mirrors DB path logic)
def _resolve_models_dir() -> Path:
    railway_data = Path("/data")
    if railway_data.exists() and railway_data.is_dir():
        d = railway_data / "run_models"
    else:
        d = Path(__file__).parent / "run_models"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        d = Path(tempfile.gettempdir()) / "19labs_run_models"
        d.mkdir(parents=True, exist_ok=True)
    return d

_MODELS_DIR = _resolve_models_dir()
print(f"[models] Storage: {_MODELS_DIR}", flush=True)

# ── DB ABSTRACTION (sqlite3 locally, psycopg2/Supabase in prod) ──────────────
def _pg_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    # Supabase/Railway give postgres:// but psycopg2 needs postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url

class _Row(dict):
    """Dict row that also supports positional integer indexing for backward compat."""
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)

class DBConn:
    """Unified DB wrapper: sqlite3 when DATABASE_URL is unset, psycopg2 (Supabase/Postgres) otherwise."""
    def __init__(self):
        url = _pg_url()
        if url:
            try:
                import psycopg2
                self._conn = psycopg2.connect(url)
                self._conn.autocommit = False
                self._cur = self._conn.cursor()
                self._pg = True
            except Exception as e:
                print(f"[db] psycopg2 connect failed ({e}), falling back to SQLite", flush=True)
                self._conn = sqlite3.connect(str(DB_PATH))
                self._conn.row_factory = sqlite3.Row
                self._cur = self._conn.cursor()
                self._pg = False
        else:
            self._conn = sqlite3.connect(str(DB_PATH))
            self._conn.row_factory = sqlite3.Row
            self._cur = self._conn.cursor()
            self._pg = False

    def execute(self, sql, params=()):
        if self._pg:
            sql = sql.replace('?', '%s')
        self._cur.execute(sql, params if params else ())
        return self

    def fetchone(self) -> "_Row | None":
        row = self._cur.fetchone()
        if row is None:
            return None
        if self._pg:
            cols = [d[0] for d in (self._cur.description or [])]
            return _Row(zip(cols, row))
        return _Row(zip(row.keys(), tuple(row)))

    def fetchall(self) -> "list[_Row]":
        rows = self._cur.fetchall()
        if self._pg:
            cols = [d[0] for d in (self._cur.description or [])]
            return [_Row(zip(cols, r)) for r in rows]
        return [_Row(zip(r.keys(), tuple(r))) for r in rows]

    def commit(self):
        self._conn.commit()
        return self

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

def _init_db():
    conn = DBConn()
    _blob_type = "BYTEA" if conn._pg else "BLOB"
    conn.execute("""CREATE TABLE IF NOT EXISTS app_config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trial_usage (
        ip TEXT NOT NULL,
        day TEXT NOT NULL,
        count INTEGER DEFAULT 0,
        PRIMARY KEY (ip, day)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS guest_runs (
        ip TEXT PRIMARY KEY,
        count INTEGER DEFAULT 0
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        user_id TEXT,
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
    # Add user_id column to existing runs tables that predate this migration
    try:
        conn.execute("ALTER TABLE runs ADD COLUMN user_id TEXT")
        conn.commit()
    except Exception:
        try:
            conn._conn.rollback()  # PostgreSQL: reset aborted transaction state
        except Exception:
            pass  # SQLite or already clean
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
    conn.execute(f"""CREATE TABLE IF NOT EXISTS run_models (
        run_id TEXT PRIMARY KEY,
        model_data {_blob_type} NOT NULL,
        model_ext TEXT DEFAULT '.pkl',
        created REAL
    )""")
    conn.commit()
    conn.close()

# ── AUTH HELPERS ──────────────────────────────────────────────
# JWT secret strategy (priority order):
#   1. JWT_SECRET env var (explicit override)
#   2. Derived from API key env vars (deterministic, survives restarts with no extra config)
#   3. Persisted in SQLite app_config (survives restarts when DB is on a persistent volume)
#   4. Random (dev only — tokens invalidated on restart, warns loudly)
def _get_or_create_jwt_secret() -> str:
    import sys as _sys
    for var in ("JWT_SECRET", "ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID", "OPENAI_API_KEY", "ACCESS_PASSWORD"):
        val = os.environ.get(var, "").strip()
        if val:
            secret = hashlib.sha256(f"19labs-jwt-v1:{val}".encode()).hexdigest()
            print(f"[auth] JWT secret derived from {var}", file=_sys.stderr, flush=True)
            return secret
    # Try to load/persist in DB
    try:
        conn = DBConn()
        row = conn.execute("SELECT value FROM app_config WHERE key='jwt_secret'").fetchone()
        if row and row[0]:
            conn.close()
            print("[auth] JWT secret loaded from database", file=_sys.stderr, flush=True)
            return row[0]
        secret = _secrets.token_hex(32)
        conn.execute(
            "INSERT INTO app_config (key, value) VALUES ('jwt_secret', ?) "
            "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value", (secret,))
        conn.commit()
        conn.close()
        print("[auth] JWT secret generated and stored in database", file=_sys.stderr, flush=True)
        return secret
    except Exception:
        pass
    print("[auth] WARNING: no stable JWT secret source — tokens will be invalidated on restart", file=_sys.stderr, flush=True)
    return _secrets.token_hex(32)

_JWT_SECRET = _get_or_create_jwt_secret()

def _make_jwt(user_id: str, email: str, name: str) -> str:
    h = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
    p = base64.urlsafe_b64encode(json.dumps({"sub": user_id, "email": email, "name": name, "exp": int(time.time()) + 365*24*3600}).encode()).rstrip(b'=').decode()
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
        conn = DBConn()
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
    if auth:
        return auth.replace("Bearer ", "").strip()
    return request.query_params.get("token", "")

def _get_user_api_key(user_id: str, provider: str) -> str:
    try:
        conn = DBConn()
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
    if provider == "bedrock":
        ak = os.environ.get("AWS_ACCESS_KEY_ID", "")
        sk = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        if ak and sk:
            return json.dumps({"access_key": ak, "secret_key": sk, "region": region})
        return ""
    _env_fallbacks = {
        "claude": os.environ.get("ANTHROPIC_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "gemini": os.environ.get("GEMINI_API_KEY", ""),
    }
    return _env_fallbacks.get(provider, "")

_init_db()

# Standalone migration: ensure run_models table exists (handles DBs initialized before this table was added)
try:
    _mc = DBConn()
    _blob_t = "BYTEA" if _mc._pg else "BLOB"
    _mc.execute(f"""CREATE TABLE IF NOT EXISTS run_models (
        run_id TEXT PRIMARY KEY,
        model_data {_blob_t} NOT NULL,
        model_ext TEXT DEFAULT '.pkl',
        created REAL
    )""")
    _mc.commit()
    _mc.close()
except Exception as _mce:
    print(f"[models] migration warning: {_mce}", flush=True)

def _save_run_to_db(run_id: str, run: dict):
    """Persist run summary (and full result JSON) to DB."""
    try:
        result = run.get("result") or {}
        best = result.get("best") or {}
        tu = result.get("token_usage") or {}
        diag = result.get("diagnostics") or {}
        # Serialize result_json — strip CSV/binary blobs, cap at 4 MB
        result_json_str = None
        if result:
            try:
                serialized = json.dumps(result)
                if len(serialized) <= 4 * 1024 * 1024:  # 4 MB cap
                    result_json_str = serialized
                else:
                    # Strip large fields and try again
                    slim = {k: v for k, v in result.items() if k not in ("plots", "csv_data")}
                    slim_serialized = json.dumps(slim)
                    if len(slim_serialized) <= 4 * 1024 * 1024:
                        result_json_str = slim_serialized
            except Exception:
                pass
        conn = DBConn()
        conn.execute("""INSERT INTO runs
            (id, user_id, status, filename, provider, started, finished, hint, budget,
             best_model, best_metric_name, best_metric_val, total_experiments,
             token_input, token_output, token_calls, yc_score, error, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
              status=EXCLUDED.status, finished=EXCLUDED.finished,
              best_model=EXCLUDED.best_model, best_metric_name=EXCLUDED.best_metric_name,
              best_metric_val=EXCLUDED.best_metric_val, total_experiments=EXCLUDED.total_experiments,
              token_input=EXCLUDED.token_input, token_output=EXCLUDED.token_output,
              token_calls=EXCLUDED.token_calls, yc_score=EXCLUDED.yc_score,
              error=EXCLUDED.error, result_json=EXCLUDED.result_json""",
            (run_id, run.get("owner_id"),
             run.get("status", "done"),
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
             result_json_str))
        conn.commit()
        conn.close()
    except Exception:
        pass  # Non-critical -- don't block the run

def _load_run_history_from_db(user_id: str | None = None):
    """Load recent runs from DB for the /api/runs endpoint.
    Authenticated users see only their own runs.
    Anonymous users see only ownerless (guest) runs."""
    try:
        conn = DBConn()
        if user_id:
            rows = conn.execute(
                "SELECT * FROM runs WHERE user_id=? ORDER BY started DESC LIMIT 100",
                (user_id,)
            ).fetchall()
        else:
            # Guest — only show runs explicitly created by guests (id starts with 'guest-').
            # Old runs with user_id IS NULL may belong to auth users created before the
            # user_id column existed, so we exclude them to prevent cross-user leakage.
            rows = conn.execute(
                "SELECT * FROM runs WHERE user_id IS NULL AND id LIKE 'guest-%' ORDER BY started DESC LIMIT 50"
            ).fetchall()
        conn.close()
        return list(rows)
    except Exception:
        return []

ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
if ALLOWED_ORIGINS == ["*"]:
    ALLOWED_ORIGINS = ["*"]

ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "").strip()

# Guest limits: 1 run total, max 4 experiments per run
GUEST_MAX_RUNS = 1
GUEST_MAX_BUDGET = 4

def _server_has_bedrock() -> bool:
    return bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))

def _trial_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    return (forwarded.split(",")[0] or request.client.host or "unknown").strip()

def _trial_check_and_increment(ip: str) -> tuple[bool, int]:
    """Returns (allowed, total_runs_used). Increments guest_runs counter if allowed.
    Guests get exactly GUEST_MAX_RUNS runs total (not per-day)."""
    try:
        conn = DBConn()
        row = conn.execute("SELECT count FROM guest_runs WHERE ip=?", (ip,)).fetchone()
        used = row[0] if row else 0
        if used >= GUEST_MAX_RUNS:
            conn.close()
            return False, used
        conn.execute(
            "INSERT INTO guest_runs (ip, count) VALUES (?,1) ON CONFLICT(ip) DO UPDATE SET count=count+1",
            (ip,)
        )
        conn.commit()
        conn.close()
        return True, used + 1
    except Exception:
        return True, 0  # fail open — don't block on DB error

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
    # Also accept a valid user JWT passed as ?token= (for direct <img src> artifact loads)
    if token and _verify_jwt(token):
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
@app.get("/ghost.svg")
def serve_ghost_svg():
    p = Path(__file__).parent / "ghost.svg"
    if p.exists():
        return FileResponse(str(p), media_type="image/svg+xml")
    return HTMLResponse("not found", status_code=404)

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

_APP_URL = "https://www.yc-able.com/app"

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

_GH_REPO = "jargogn0/yc-able"
_GH_RELEASE_BASE = f"https://github.com/{_GH_REPO}/releases"
_APP_VERSION = "1.0.0"

def _gh_release_url(version: str, filename: str) -> str:
    return f"{_GH_RELEASE_BASE}/download/v{version}/{filename}"

def _mac_dmg_url(version: str, arch: str = "x64") -> str:
    # electron-builder naming: 19Labs-1.0.0.dmg (x64), 19Labs-1.0.0-arm64.dmg (arm64)
    if arch == "arm64":
        return _gh_release_url(version, f"19Labs-{version}-arm64.dmg")
    return _gh_release_url(version, f"19Labs-{version}.dmg")

@app.get("/api/latest-release")
def latest_release():
    return {"version": f"v{_APP_VERSION}", "mac_x64": _mac_dmg_url(_APP_VERSION), "mac_arm64": _mac_dmg_url(_APP_VERSION, "arm64")}

@app.get("/download")
async def download_auto(request: Request):
    ua = request.headers.get("user-agent", "").lower()
    if "windows" in ua or "win64" in ua or "win32" in ua:
        return RedirectResponse(_gh_release_url(_APP_VERSION, f"19Labs-Setup-{_APP_VERSION}.exe"), status_code=302)
    if "mac" in ua or "darwin" in ua:
        # Default to x64 (works on Apple Silicon via Rosetta too)
        return RedirectResponse(_mac_dmg_url(_APP_VERSION), status_code=302)
    return RedirectResponse(_gh_release_url(_APP_VERSION, f"19Labs-{_APP_VERSION}.AppImage"), status_code=302)

@app.get("/download/mac")
async def _download_mac(request: Request, arch: str = ""):
    """Download macOS DMG. ?arch=arm64 for Apple Silicon native build."""
    a = arch or ("arm64" if "arm64" in request.headers.get("user-agent", "").lower() else "x64")
    return RedirectResponse(_mac_dmg_url(_APP_VERSION, a), status_code=302)

@app.get("/download/mac/arm64")
def _download_mac_arm64():
    return RedirectResponse(_mac_dmg_url(_APP_VERSION, "arm64"), status_code=302)

@app.get("/download/win")
def _download_windows():
    return RedirectResponse(_gh_release_url(_APP_VERSION, f"19Labs-Setup-{_APP_VERSION}.exe"), status_code=302)

@app.get("/download/linux")
def _download_linux():
    return RedirectResponse(_gh_release_url(_APP_VERSION, f"19Labs-{_APP_VERSION}.AppImage"), status_code=302)

# ── AUTH ENDPOINTS ────────────────────────────────────────────
# In-memory CSRF state store for Google OAuth (short-lived)
_OAUTH_STATES: dict[str, float] = {}

def _site_base(request: Request) -> str:
    """Return the public base URL, always https in production.
    Reads SITE_URL env var first (e.g. https://19labs.dev),
    then falls back to request.base_url with http→https forced."""
    site = os.environ.get("SITE_URL", "").strip().rstrip("/")
    if site:
        return site
    base = _site_base(request)
    # Railway / any reverse-proxy: request arrives as http:// internally
    if base.startswith("http://"):
        base = "https://" + base[7:]
    return base

def _validate_provider_key(provider: str, api_key: str) -> tuple[bool, str]:
    """Validate an API key by calling the provider. Returns (valid, display_name)."""
    try:
        if provider in ("claude", "anthropic"):
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                return r.status == 200, "Anthropic"
        elif provider == "openai":
            req = urllib.request.Request(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                return r.status == 200, "OpenAI"
        elif provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            with urllib.request.urlopen(url, timeout=8) as r:
                return r.status == 200, "Google Gemini"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, ""
        # Other HTTP errors (rate limit, etc.) — assume key format is okay
        return True, provider.title()
    except Exception:
        # Network issues — optimistically accept; will fail at run time if truly invalid
        return True, provider.title()
    return False, ""

def _upsert_provider_user(user_id: str, email: str, name: str, provider: str, api_key: str):
    """Create or update a user record and save their API key."""
    conn = DBConn()
    conn.execute(
        "INSERT INTO users (id, email, name, password_hash, created) VALUES (?,?,?,?,?) "
        "ON CONFLICT (id) DO UPDATE SET email=EXCLUDED.email, name=EXCLUDED.name",
        (user_id, email, name, "", time.time())
    )
    norm = "claude" if provider == "anthropic" else provider
    conn.execute(
        "INSERT INTO user_api_keys (user_id, provider, api_key) VALUES (?,?,?) "
        "ON CONFLICT (user_id, provider) DO UPDATE SET api_key=EXCLUDED.api_key",
        (user_id, norm, api_key)
    )
    conn.commit()
    conn.close()

@app.post("/auth/provider-login")
async def provider_login(request: Request):
    """Sign in / sign up using an AI provider API key. The key is the credential."""
    data = await request.json()
    provider = (data.get("provider") or "").strip().lower()
    api_key = (data.get("api_key") or "").strip()
    if provider not in ("claude", "anthropic", "openai", "gemini"):
        raise HTTPException(400, "provider must be claude, openai, or gemini")
    if not api_key:
        raise HTTPException(400, "api_key is required")

    valid, display = _validate_provider_key(provider, api_key)
    if not valid:
        raise HTTPException(401, "API key rejected by the provider. Check it and try again.")

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:20]
    norm = "claude" if provider == "anthropic" else provider
    user_id = f"{norm}:{key_hash}"
    email = f"{key_hash[:10]}@{norm}.key"
    name = display or norm.title()

    _upsert_provider_user(user_id, email, name, norm, api_key)
    token = _make_jwt(user_id, email, name)
    return {"token": token, "user": {"id": user_id, "email": email, "name": name, "provider": norm}}

@app.get("/auth/google")
async def google_auth_start(request: Request):
    """Redirect to Google OAuth. Requires GOOGLE_CLIENT_ID env var."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        raise HTTPException(501, "Google sign-in is not configured on this server.")
    state = _secrets.token_hex(16)
    _OAUTH_STATES[state] = time.time()
    # Clean up old states (>10 min)
    old = [k for k, t in _OAUTH_STATES.items() if time.time() - t > 600]
    for k in old:
        del _OAUTH_STATES[k]
    base = _site_base(request)
    redirect_uri = urllib.parse.quote(f"{base}/auth/google/callback", safe="")
    scope = urllib.parse.quote("openid email profile", safe="")
    url = (f"https://accounts.google.com/o/oauth2/v2/auth"
           f"?client_id={client_id}&redirect_uri={redirect_uri}"
           f"&response_type=code&scope={scope}&state={state}&prompt=select_account")
    return RedirectResponse(url, status_code=302)

@app.get("/auth/github")
async def github_auth_start(request: Request):
    """Redirect to GitHub OAuth. Requires GITHUB_CLIENT_ID env var."""
    client_id = os.environ.get("GITHUB_CLIENT_ID", "").strip()
    if not client_id:
        raise HTTPException(501, "GitHub sign-in is not configured on this server.")
    state = _secrets.token_hex(16)
    _OAUTH_STATES[state] = time.time()
    old = [k for k, t in _OAUTH_STATES.items() if time.time() - t > 600]
    for k in old:
        del _OAUTH_STATES[k]
    base = _site_base(request)
    redirect_uri = urllib.parse.quote(f"{base}/auth/github/callback", safe="")
    url = (f"https://github.com/login/oauth/authorize"
           f"?client_id={client_id}&redirect_uri={redirect_uri}"
           f"&scope=user:email&state={state}")
    return RedirectResponse(url, status_code=302)

@app.get("/auth/github/callback")
async def github_auth_callback(code: str = "", state: str = "", error: str = "", request: Request = None):
    """Handle GitHub OAuth callback."""
    if error or not code:
        return HTMLResponse("<script>window.location='/app?auth_error=cancelled'</script>")
    if state not in _OAUTH_STATES:
        return HTMLResponse("<script>window.location='/app?auth_error=invalid_state'</script>")
    del _OAUTH_STATES[state]

    client_id = os.environ.get("GITHUB_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GITHUB_CLIENT_SECRET", "").strip()
    base = _site_base(request)
    redirect_uri = f"{base}/auth/github/callback"

    try:
        # Exchange code → access token
        token_data = urllib.parse.urlencode({
            "code": code, "client_id": client_id, "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        }).encode()
        req = urllib.request.Request("https://github.com/login/oauth/access_token", data=token_data,
                                     headers={"Content-Type": "application/x-www-form-urlencoded",
                                              "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            token_resp = json.loads(r.read())

        access_token = token_resp.get("access_token", "")
        if not access_token:
            raise ValueError(f"No access token returned: {token_resp}")

        # Fetch user info
        ui_req = urllib.request.Request("https://api.github.com/user",
                                        headers={"Authorization": f"Bearer {access_token}",
                                                 "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(ui_req, timeout=10) as r:
            info = json.loads(r.read())

        # GitHub may not expose email — fetch emails list
        email = info.get("email") or ""
        if not email:
            em_req = urllib.request.Request("https://api.github.com/user/emails",
                                            headers={"Authorization": f"Bearer {access_token}",
                                                     "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(em_req, timeout=10) as r:
                emails = json.loads(r.read())
            primary = next((e["email"] for e in emails if e.get("primary") and e.get("verified")), None)
            email = primary or (emails[0]["email"] if emails else f"gh_{info['id']}@github.local")

        name = info.get("name") or info.get("login") or email.split("@")[0]
        user_id = f"github:{info['id']}"

        conn = DBConn()
        existing = conn.execute("SELECT id, name FROM users WHERE email=?", (email,)).fetchone()
        if existing:
            user_id = existing["id"]
            name = existing["name"] or name
        else:
            conn.execute(
                "INSERT INTO users (id, email, name, password_hash, created) VALUES (?,?,?,?,?) "
                "ON CONFLICT (id) DO UPDATE SET email=EXCLUDED.email, name=EXCLUDED.name",
                (user_id, email, name, "", time.time()))
        conn.commit()
        conn.close()

        token = _make_jwt(user_id, email, name)
        safe_token = token.replace("'", "")
        return HTMLResponse(
            f"<script>localStorage.setItem('19labs_auth_token','{safe_token}');"
            f"window.location.href='/app';</script>"
        )
    except Exception as e:
        return HTMLResponse(f"<script>window.location='/app?auth_error={urllib.parse.quote(str(e))}'</script>")

@app.get("/auth/google/callback")
async def google_auth_callback(code: str = "", state: str = "", error: str = "", request: Request = None):
    """Handle Google OAuth callback."""
    if error or not code:
        return HTMLResponse("<script>window.location='/app?auth_error=cancelled'</script>")
    if state not in _OAUTH_STATES:
        return HTMLResponse("<script>window.location='/app?auth_error=invalid_state'</script>")
    del _OAUTH_STATES[state]

    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    base = _site_base(request)
    redirect_uri = f"{base}/auth/google/callback"

    try:
        # Exchange code → tokens
        token_data = urllib.parse.urlencode({
            "code": code, "client_id": client_id, "client_secret": client_secret,
            "redirect_uri": redirect_uri, "grant_type": "authorization_code"
        }).encode()
        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=token_data,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=10) as r:
            tokens = json.loads(r.read())

        # Fetch user info
        ui_req = urllib.request.Request("https://www.googleapis.com/oauth2/v3/userinfo",
                                        headers={"Authorization": f"Bearer {tokens['access_token']}"})
        with urllib.request.urlopen(ui_req, timeout=10) as r:
            info = json.loads(r.read())

        email = info.get("email", "")
        name = info.get("name", email.split("@")[0])
        user_id = f"google:{info.get('sub', hashlib.sha256(email.encode()).hexdigest()[:16])}"

        conn = DBConn()
        existing = conn.execute("SELECT id, name FROM users WHERE email=?", (email,)).fetchone()
        if existing:
            # Email already registered (e.g. via email/password) — log in as that account
            user_id = existing["id"]
            name = existing["name"] or name
        else:
            conn.execute(
                "INSERT INTO users (id, email, name, password_hash, created) VALUES (?,?,?,?,?) "
                "ON CONFLICT (id) DO UPDATE SET email=EXCLUDED.email, name=EXCLUDED.name",
                (user_id, email, name, "", time.time()))
        conn.commit()
        conn.close()

        token = _make_jwt(user_id, email, name)
        safe_token = token.replace("'", "")
        return HTMLResponse(
            f"<script>localStorage.setItem('19labs_auth_token','{safe_token}');"
            f"window.location.href='/app';</script>"
        )
    except Exception as e:
        return HTMLResponse(f"<script>window.location='/app?auth_error={urllib.parse.quote(str(e))}'</script>")

def _email_user_id(email: str) -> str:
    """Deterministic user ID from email — survives DB wipes."""
    return "u:" + hashlib.sha256(email.lower().strip().encode()).hexdigest()[:24]

@app.post("/auth/signup")
async def auth_signup(request: Request):
    data = await request.json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name = (data.get("name", "") or email.split("@")[0]).strip()
    if not email or "@" not in email:
        raise HTTPException(400, "Invalid email address")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    # Deterministic user_id — same email always maps to the same ID even after DB wipe
    user_id = _email_user_id(email)
    conn = DBConn()
    row = conn.execute("SELECT id, email, name, password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    if row:
        # Account exists — treat signup as login attempt (idempotent)
        if _verify_pw(password, row[3]):
            conn.close()
            token = _make_jwt(row[0], row[1], row[2])
            return {"token": token, "user": {"id": row[0], "email": row[1], "name": row[2]}}
        conn.close()
        raise HTTPException(409, "An account with this email already exists. Please sign in with your existing password.")
    conn.execute("INSERT INTO users (id, email, name, password_hash, created) VALUES (?,?,?,?,?)",
                 (user_id, email, name, _hash_pw(password), time.time()))
    conn.commit()
    conn.close()
    token = _make_jwt(user_id, email, name)
    return {"token": token, "user": {"id": user_id, "email": email, "name": name}}

@app.post("/auth/login")
async def auth_login(request: Request):
    data = await request.json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not email or not password:
        raise HTTPException(400, "Email and password required")
    user_id = _email_user_id(email)
    conn = DBConn()
    row = conn.execute("SELECT id, email, name, password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(401, "No account found with this email. If you had an account before, the server may have reset — please create a new account (your API keys are saved in your browser).")
    if not _verify_pw(password, row[3]):
        raise HTTPException(401, "Incorrect password. Please check and try again.")
    token = _make_jwt(row[0], row[1], row[2])
    return {"token": token, "user": {"id": row[0], "email": row[1], "name": row[2]}}

@app.get("/auth/me")
async def auth_me(request: Request):
    user = _get_user(_token_from_request(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    return user

@app.post("/auth/refresh")
async def auth_refresh(request: Request):
    """Issue a fresh token for a still-valid session (extends expiry)."""
    user = _get_user(_token_from_request(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    token = _make_jwt(user["id"], user["email"], user.get("name", ""))
    return {"token": token, "user": user}

@app.post("/auth/logout")
async def auth_logout(request: Request):
    token = _token_from_request(request)
    if token:
        conn = DBConn()
        conn.execute("DELETE FROM user_sessions WHERE token=?", (token,))
        conn.commit()
        conn.close()
    return {"ok": True}

@app.get("/auth/keys")
async def auth_get_keys(request: Request):
    user = _get_user(_token_from_request(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    conn = DBConn()
    rows = conn.execute(
        "SELECT provider, api_key FROM user_api_keys WHERE user_id=?", (user["id"],)
    ).fetchall()
    conn.close()
    result = {}
    for row in rows:
        p, k = row['provider'], row['api_key']
        result[p] = (k[:8] + "…" + k[-4:]) if len(k) > 12 else "****"
    return result

@app.post("/auth/keys")
async def auth_save_keys(request: Request):
    user = _get_user(_token_from_request(request))
    if not user:
        raise HTTPException(401, "Not authenticated")
    data = await request.json()
    conn = DBConn()
    for provider, key in data.items():
        # normalize: 'anthropic' → 'claude' to match engine provider names
        if provider == "anthropic":
            provider = "claude"
        if provider == "bedrock":
            if isinstance(key, dict):
                key = json.dumps(key)
            elif not isinstance(key, str):
                continue
        if provider in ("claude", "openai", "gemini", "bedrock") and key and (isinstance(key, str) and key.strip()):
            conn.execute(
                "INSERT INTO user_api_keys (user_id, provider, api_key) VALUES (?,?,?) "
                "ON CONFLICT (user_id, provider) DO UPDATE SET api_key=EXCLUDED.api_key",
                (user["id"], provider, key.strip() if isinstance(key, str) else key)
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
    model: str = ""        # optional model override (e.g. "gpt-4o-mini", "claude-opus-4-6")
    continuous: bool = False

class ValidateKeyRequest(BaseModel):
    api_key: str = ""
    provider: str = "claude"

class DiscoverRequest(BaseModel):
    filename: str
    csv: str = ""
    dataset_id: str = ""   # pre-uploaded media dataset
    hint: str = ""
    previous_objective: dict | None = None  # what the agent was proposing before the user corrected it
    api_key: str = ""
    provider: str = "claude"
    model: str = ""        # optional model override

class ChatRequest(BaseModel):
    message: str
    api_key: str = ""
    provider: str = "claude"
    model: str = ""        # optional model override
    context: dict = {}

class PredictRequest(BaseModel):
    data: list[dict] = []  # list of rows as dicts
    csv_text: str = ""     # alternative: raw CSV text

def _get_run_or_404(run_id: str):
    """Return run dict from memory or reconstruct from DB. Raises HTTPException if not found."""
    if run_id in RUNS:
        return RUNS[run_id]
    try:
        conn = DBConn()
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        conn.close()
        if row and row["result_json"]:
            result = json.loads(row["result_json"])
            # Try to recover workspace path from /tmp glob (still valid within 4-hour window)
            import glob as _glob
            _ws_candidates = _glob.glob(str(Path(tempfile.gettempdir()) / f"19labs_{run_id}_*"))
            _ws_str = _ws_candidates[0] if _ws_candidates else ""
            run = {
                "id": run_id,
                "status": row["status"],
                "ws": _ws_str,
                "result": result,
                "hint": row.get("hint", ""),
            }
            return run
    except Exception:
        pass
    raise HTTPException(404, "Run not found")


def _load_autogluon_predictor(ag_path: str):
    """Load an AutoGluon TabularPredictor from its saved directory."""
    from autogluon.tabular import TabularPredictor as _AGP
    return _AGP.load(ag_path, require_version_match=False)


def _find_autogluon_path(run_id: str, run: dict) -> "str | None":
    """Return the persistent AutoGluon model directory path if available."""
    result_d = run.get("result") or {}
    # Check result for stored ag_path
    if result_d.get("model_ag_path"):
        _p = Path(result_d["model_ag_path"])
        if _p.exists() and any(_p.rglob("*")):
            return str(_p)
    # Check persistent model dir
    _stable = _MODELS_DIR / f"{run_id}_ag"
    if _stable.exists() and any(_stable.rglob("*")):
        return str(_stable)
    # Check if model.pkl is an AG reference dict
    _pkl = _MODELS_DIR / f"{run_id}.pkl"
    if _pkl.exists():
        try:
            import joblib as _jl
            _d = _jl.load(_pkl)
            if isinstance(_d, dict) and _d.get("type") == "autogluon":
                _ap = _d.get("ag_path", "")
                if _ap and Path(_ap).exists():
                    return _ap
        except Exception:
            pass
    return None


def _find_model_path(run_id: str, run: dict) -> "Path | None":
    """Find model file — checks result_json base64, filesystem, then DB BLOB."""
    # 0. Best: model_b64 embedded in result_json — only use if it's a binary model format
    result_d = run.get("result") or {}
    if result_d.get("model_b64"):
        _ext = result_d.get("model_ext", ".pkl")
        if _ext in (".pkl", ".joblib", ".ubj"):  # never trust .json from old buggy code
            try:
                import base64 as _b64
                _tmp = Path(tempfile.gettempdir()) / f"19labs_model_{run_id}{_ext}"
                _tmp.write_bytes(_b64.b64decode(result_d["model_b64"]))
                print(f"[predict] loaded model from result_json base64 → {_tmp}", flush=True)
                return _tmp
            except Exception as _b64e:
                print(f"[predict] model_b64 decode failed: {_b64e}", flush=True)
    # 1. Persistent stable path (survives restarts)
    for ext in [".pkl", ".joblib"]:
        stable = _MODELS_DIR / f"{run_id}{ext}"
        if stable.exists():
            return stable
    # 2. Workspace search (valid within same server session / 4-hour window)
    ws_str = run.get("ws", "")
    ws = Path(ws_str) if ws_str else None
    result = run.get("result") or {}
    deploy_path = result.get("deploy_path")
    dp = Path(deploy_path) if deploy_path and Path(deploy_path).exists() else None
    # Also try recovering workspace from /tmp glob (handles multi-instance / post-restart)
    if not ws:
        import glob as _g2
        _cands = _g2.glob(str(Path(tempfile.gettempdir()) / f"19labs_{run_id}_*"))
        ws = Path(_cands[0]) if _cands else None
    search_dirs = [d for d in [dp, ws] if d and str(d) and d.exists()]
    print(f"[predict] model search for {run_id}: MODELS_DIR={_MODELS_DIR}, dp={dp}, ws={ws}, dirs={search_dirs}", flush=True)
    # Explicit common names first
    for name in ["best_model.pkl", "model.pkl", "best_model.joblib", "model.joblib"]:
        for d in search_dirs:
            f = d / name
            if f.is_file():
                print(f"[predict] found model: {f}", flush=True)
                return f
    # Full recursive fallback — pkl/joblib/ubj only; JSON files are partial components
    for d in search_dirs:
        all_files = list(d.rglob("*.pkl")) + list(d.rglob("*.joblib")) + list(d.rglob("*.ubj"))
        print(f"[predict] rglob in {d}: {[str(f) for f in all_files[:10]]}", flush=True)
        for f in all_files:
            if f.is_file() and "api_server" not in str(f) and "sample_sub" not in str(f.name):
                print(f"[predict] found model via rglob: {f}", flush=True)
                return f
    # 3. SQLite BLOB fallback — works even after container restart wipes /tmp
    _db_status = "not_tried"
    try:
        _conn = DBConn()
        _row = _conn.execute("SELECT model_data, model_ext FROM run_models WHERE run_id=?", (run_id,)).fetchone()
        _conn.close()
        if _row and _row["model_data"]:
            _tmp_path = Path(tempfile.gettempdir()) / f"19labs_model_{run_id}{_row['model_ext']}"
            _tmp_path.write_bytes(bytes(_row["model_data"]))
            print(f"[predict] loaded model from DB BLOB → {_tmp_path}", flush=True)
            return _tmp_path
        _db_status = f"row_found={_row is not None},has_data={bool(_row and _row['model_data']) if _row else False}"
    except Exception as _dbe:
        _db_status = f"error:{_dbe}"
        print(f"[predict] DB model lookup failed: {_dbe}", flush=True)
    print(f"[predict] WARNING: no model found for {run_id} | db_status={_db_status}", flush=True)
    return None


def _fallback_discovery(profile: dict, hint: str = ""):
    headers = profile.get("headers", [])
    numeric = profile.get("numeric", [])
    categorical = profile.get("categorical", [])
    target_candidates = profile.get("target_candidates", [])
    target = target_candidates[0] if target_candidates else (numeric[-1] if numeric else (headers[-1] if headers else "target"))
    rows = profile.get("rows", 0)
    cols = profile.get("cols", 0)
    signals = profile.get("signals", [])
    class_balance = profile.get("class_balance", {})

    # Detect task type from target column info
    target_col = next((c for c in profile.get("columns", []) if c["name"] == target), None)
    is_classification = target_col and target_col.get("type") in ("categorical", "high_cardinality") and target_col.get("unique", 99) <= 20
    task = "BinaryClassification" if (is_classification and target_col and target_col.get("unique", 0) <= 2) else \
           "MultiClassClassification" if is_classification else "Regression"
    metric = "auc" if "classification" in task.lower() else "rmse"

    # Detect Kaggle competition from hint
    _hint_lo = (hint or "").lower()
    _is_comp = "competition" in _hint_lo or "kaggle" in _hint_lo

    # Build a smart, signal-driven message from the actual data
    _balance_pct = 0
    _majority_class = ""
    if target in class_balance:
        _bal_items = list(class_balance[target].items())
        if _bal_items:
            _majority_class, _majority_val = max(_bal_items, key=lambda x: x[1])
            _balance_pct = _majority_val

    _has_datetime = any("DATETIME" in s for s in signals)
    _has_imbalance = _balance_pct > 0.7 or any("IMBALANCED" in s for s in signals)
    _high_dim = cols > 50
    _large = rows > 100_000

    # Pick the single most notable data characteristic as the opening hook
    if _has_imbalance and _balance_pct > 0:
        _hook = f"'{target}' is {_balance_pct*100:.0f}% '{_majority_class}' — heavy class imbalance, so AUC with stratified folds is the way to go."
    elif _has_datetime:
        _hook = f"There's a datetime column in here — this has time-series structure, so order matters for validation."
    elif _high_dim:
        _hook = f"{cols} features is a lot — feature selection and regularization will matter to avoid overfitting."
    elif _large:
        _hook = f"{rows:,} rows is solid — enough data to be confident in CV estimates without too much overfitting risk."
    elif numeric and len(numeric) > len(categorical):
        _hook = f"Mostly numeric features ({len(numeric)} of them) targeting '{target}' — clean setup for gradient boosting."
    elif categorical:
        _hook = f"{len(categorical)} categorical features — encoding strategy will matter here, especially for high-cardinality ones."
    else:
        _hook = f"Predicting '{target}' from {cols} features across {rows:,} rows."

    _task_label = "binary classification" if "binary" in task.lower() else ("multi-class classification" if "multi" in task.lower() else "regression")
    _model_hint = "LightGBM + class weights" if _has_imbalance and "classif" in task.lower() else \
                  "LightGBM with time-based split" if _has_datetime else \
                  "LightGBM baseline"
    if _is_comp:
        _agent_msg = (
            f"Competition mode — predicting **{target}** ({_task_label}, {metric.upper()}). "
            f"{_hook} "
            f"Starting with {_model_hint}."
        )
    else:
        _agent_msg = (
            f"Predicting **{target}** — {_task_label}, {metric.upper()} metric. "
            f"{_hook} "
            f"Starting with {_model_hint}."
        ).replace("  ", " ").strip()

    objective = {
        "domain": "General",
        "task": task,
        "target": target,
        "metric": metric,
        "direction": "lower_is_better" if metric == "rmse" else "higher_is_better",
        "confidence": 0.45,
        "reasoning": "Fallback inference used because AI discovery was unavailable.",
        "good_enough": "",
        "raw": "fallback",
    }
    discovery = {
        "recommended_objective": f"Predict {target} from the available features.",
        "recommended_metric": metric,
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
        "discovery": {**discovery, "agent_message": _agent_msg},
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


@app.get("/api/config")
async def get_config():
    """Public config — tells the frontend what's available server-side."""
    return {
        "has_bedrock": _server_has_bedrock(),
        "guest_max_runs": GUEST_MAX_RUNS,
        "guest_max_budget": GUEST_MAX_BUDGET,
        "bedrock_model": os.environ.get("BEDROCK_MODEL", "anthropic.claude-sonnet-4-6"),
    }

@app.post("/api/upload-dataset")
async def upload_media_dataset(file: UploadFile = File(...)):
    """Accept ZIP/image/audio/CSV dataset. Returns dataset_id for use in /api/run."""
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
            # Security: skip absolute paths and path traversal entries
            for member in zf.infolist():
                member_path = Path(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                zf.extract(member, data_dir)
        zip_path.unlink()
        # Unwrap single top-level folder if that's all there is
        items = [x for x in data_dir.iterdir()]
        if len(items) == 1 and items[0].is_dir():
            data_dir = items[0]

        # ── Check for tabular files inside the ZIP ──────────────────
        _TABULAR_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}
        tabular_files = [
            f for f in data_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in _TABULAR_EXTS and not f.name.startswith(".")
        ]
        if tabular_files:
            # Prefer train.csv / train*.csv as primary; fall back to largest file
            def _primary_rank(f):
                n = f.name.lower()
                if n == "train.csv": return 0
                if n.startswith("train"): return 1
                return 2
            tabular_files.sort(key=lambda f: (_primary_rank(f), -f.stat().st_size))
            primary = tabular_files[0]
            try:
                # Convert non-CSV formats to CSV in-place
                if primary.suffix.lower() in (".xlsx", ".xls"):
                    import pandas as _pd
                    csv_path_conv = primary.with_suffix(".csv")
                    _pd.read_excel(primary).to_csv(csv_path_conv, index=False)
                    primary = csv_path_conv
                elif primary.suffix.lower() == ".parquet":
                    import pandas as _pd
                    csv_path_conv = primary.with_suffix(".csv")
                    _pd.read_parquet(primary).to_csv(csv_path_conv, index=False)
                    primary = csv_path_conv
                elif primary.suffix.lower() == ".json":
                    import pandas as _pd
                    csv_path_conv = primary.with_suffix(".csv")
                    _pd.read_json(primary).to_csv(csv_path_conv, index=False)
                    primary = csv_path_conv

                all_names = [f.name for f in tabular_files]
                is_large = primary.stat().st_size > 8 * 1024 * 1024
                is_multi = len(tabular_files) > 1

                if not is_large and not is_multi:
                    # Small single file — return inline so frontend shows a data preview table
                    csv_text = primary.read_text(errors="replace")
                    return {
                        "dataset_id": dataset_id,
                        "type": "csv",
                        "filename": primary.name,
                        "csv": csv_text,
                        "all_files": all_names,
                        "num_files": len(tabular_files),
                    }
                else:
                    # Large or multi-file (e.g. Kaggle train/test/sample) — keep server-side
                    _MEDIA_DATASETS[dataset_id] = {
                        "path": str(data_dir),
                        "type": "tabular_dir",
                        "primary": str(primary),
                        "all_files": [str(f) for f in tabular_files],
                    }
                    # Include a small preview (first 100 rows) so the frontend can show a data table
                    try:
                        import io as _io
                        import pandas as _pd_prev
                        _preview_df = _pd_prev.read_csv(primary, nrows=100)
                        _preview_buf = _io.StringIO()
                        _preview_df.to_csv(_preview_buf, index=False)
                        _preview_csv = _preview_buf.getvalue()
                    except Exception:
                        _preview_csv = ""
                    # Read sample_submission columns so frontend can show competition context
                    _samp_cols_list = []
                    for _sf in tabular_files:
                        if "sample" in _sf.name.lower() or "submission" in _sf.name.lower():
                            try:
                                _samp_cols_list = list(__import__("pandas").read_csv(_sf, nrows=1).columns)
                            except Exception:
                                pass
                            break
                    return {
                        "dataset_id": dataset_id,
                        "type": "tabular_dir",
                        "filename": primary.name,
                        "all_files": all_names,
                        "num_files": len(tabular_files),
                        "sample_submission_cols": _samp_cols_list,
                        "preview_csv": _preview_csv,
                    }
            except Exception as e:
                # Fall through to media profiling if read/convert fails
                pass
    else:
        data_dir = tmp_dir / "data"
        data_dir.mkdir()
        (data_dir / filename).write_bytes(content)

    info = _profile_media_dir(data_dir)
    _MEDIA_DATASETS[dataset_id] = {"path": str(data_dir), "filename": filename, **info}
    return {"dataset_id": dataset_id, **info}


@app.post("/api/run")
async def start_run(req: RunRequest, request: Request):
    user = _get_user(_token_from_request(request))

    # Auto-fill API key from user account if not provided
    if not req.api_key and user:
        saved = _get_user_api_key(user["id"], req.provider)
        if saved:
            req.api_key = saved

    # Guest path — 1 free run (max 4 experiments) using server Bedrock
    if not req.api_key and not user:
        if not _server_has_bedrock():
            raise HTTPException(401, "Sign in to run experiments. Create a free account.")
        ip = _trial_ip(request)
        allowed, used = _trial_check_and_increment(ip)
        if not allowed:
            raise HTTPException(429, "GUEST_LIMIT_REACHED")
        # Hard cap: guests get at most GUEST_MAX_BUDGET experiments per run
        req.budget = min(req.budget or GUEST_MAX_BUDGET, GUEST_MAX_BUDGET)

    # Build meaningful run ID prefix
    if user:
        # Auth user: initials from name or email, e.g. "dba" for "Doudou Ba"
        raw = (user.get("name") or user.get("email") or "u").strip()
        parts = raw.split()
        if len(parts) >= 2:
            prefix = (parts[0][0] + parts[-1][:2]).lower()
        else:
            prefix = raw[:3].lower()
        prefix = ''.join(c for c in prefix if c.isalnum()) or "u"
    else:
        prefix = "guest"
    run_id = f"{prefix}-{str(uuid.uuid4())[:8]}"
    ws = Path(tempfile.mkdtemp(prefix=f"19labs_{run_id}_"))

    if req.dataset_id:
        media = _MEDIA_DATASETS.get(req.dataset_id)
        if not media:
            raise HTTPException(404, "Dataset not found. Please re-upload.")
        if media.get("type") == "tabular_dir":
            # Copy all tabular files into the workspace so the engine and AI-generated code can use them
            src_dir = Path(media["path"])
            for src_str in media["all_files"]:
                src_p = Path(src_str)
                if src_p.exists():
                    shutil.copy2(src_p, ws / src_p.name)
            csv_path = ws / Path(media["primary"]).name
            # Append Kaggle context to user hint so engine detects competition automatically
            _all_names = [Path(f).name for f in media.get("all_files", [])]
            _samp = next((Path(f) for f in media.get("all_files", []) if ("sample" in Path(f).name.lower() or "submission" in Path(f).name.lower()) and Path(f).exists()), None)
            _test = next((Path(f) for f in media.get("all_files", []) if "test" in Path(f).name.lower() and Path(f).exists()), None)
            if _test or _samp:
                _kparts = [f"Files in workspace: {', '.join(_all_names)}."]
                if _test:
                    _kparts.append(f"Generate predictions for {_test.name} using the trained model.")
                if _samp:
                    try:
                        import pandas as _pd2
                        _sc = list(_pd2.read_csv(_samp, nrows=2).columns)
                        _kparts.append(f"Save submission.csv with columns {_sc} matching {_samp.name}. Target column = {_sc[-1]}.")
                    except Exception:
                        _kparts.append(f"Save submission.csv matching {_samp.name}.")
                _ctx = " [KAGGLE COMPETITION: " + " ".join(_kparts) + "]"
                req.hint = ((req.hint or "") + _ctx).strip()
        else:
            # Image / audio media dataset — pass directory to engine
            csv_path = Path(media["path"])
    else:
        # Direct CSV text upload
        csv_path = ws / req.filename
        csv_path.write_text(req.csv, encoding="utf-8")

    cancel_event = threading.Event()
    RUNS[run_id] = dict(
        id=run_id, owner_id=user["id"] if user else None,
        status="running", ws=str(ws),
        csv=str(csv_path), logs=[], result=None,
        started=time.time(), provider=req.provider,
        hint=req.hint or "", budget=req.budget,
        cancel_event=cancel_event,
    )
    # Persist immediately so the run survives a server restart
    _save_run_to_db(run_id, RUNS[run_id])

    def background():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from engine import run_research

        def cb(tag, msg):
            RUNS[run_id]["logs"].append({
                "tag": tag, "msg": msg, "ts": time.strftime("%H:%M:%S")
            })

        try:
            _prov = (req.provider or "claude").lower()
            if not req.api_key and _prov == "bedrock":
                _ak = os.environ.get("AWS_ACCESS_KEY_ID", "")
                _sk = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
                _rg = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
                resolved_api_key = json.dumps({"access_key": _ak, "secret_key": _sk, "region": _rg}) if _ak and _sk else ""
            else:
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
                model=req.model or None,
            )
            # Capture train.py content and logs into result so they survive DB persistence
            if result is None:
                result = {}
            train_py_path = ws / "train.py"
            if train_py_path.exists():
                try:
                    result["train_code"] = train_py_path.read_text()
                except Exception:
                    pass
            result["run_logs"] = [{"tag": l["tag"], "msg": l["msg"], "ts": l.get("ts", "")}
                                  for l in RUNS[run_id].get("logs", [])[-500:]]  # keep last 500
            # Record what files exist in workspace for diagnostics
            try:
                result["_ws_files"] = [f.name for f in ws.rglob("*") if f.is_file()][:40]
            except Exception:
                result["_ws_files"] = []
            RUNS[run_id]["result"] = result
            # ── Persist model ──────────────────────────────────────────────────
            # Priority 1: AutoGluon ag_models/ directory → copy to persistent storage
            _ag_ws_dir = ws / "ag_models"
            if _ag_ws_dir.exists() and any(_ag_ws_dir.rglob("*")):
                try:
                    _ag_persistent = _MODELS_DIR / f"{run_id}_ag"
                    if _ag_persistent.exists():
                        shutil.rmtree(_ag_persistent)
                    shutil.copytree(_ag_ws_dir, _ag_persistent)
                    result["model_ag_path"] = str(_ag_persistent)
                    # Update model.pkl to point to persistent path
                    _pkl_path = ws / "model.pkl"
                    import joblib as _jl
                    _jl.dump({"type": "autogluon", "ag_path": str(_ag_persistent)}, _pkl_path)
                    shutil.copy2(_pkl_path, _MODELS_DIR / f"{run_id}.pkl")
                    RUNS[run_id]["result"] = result
                    print(f"[models] AutoGluon model copied to persistent storage: {_ag_persistent}", flush=True)
                except Exception as _age:
                    print(f"[models] AutoGluon persist failed: {_age}", flush=True)

            # Priority 2: standard pkl/joblib — embed as base64 in result_json
            _model_src = None
            for _mname in ["best_model.pkl", "model.pkl", "best_model.joblib", "model.joblib"]:
                _mp = ws / _mname
                if _mp.exists():
                    _model_src = _mp
                    break
            if not _model_src:
                for _mp in list(ws.rglob("*.pkl")) + list(ws.rglob("*.joblib")) + list(ws.rglob("*.ubj")):
                    if _mp.is_file() and "sample" not in _mp.name and "ag_models" not in str(_mp):
                        _model_src = _mp
                        break
            if _model_src:
                try:
                    import base64 as _b64
                    _blob = _model_src.read_bytes()
                    # Don't embed huge AutoGluon reference files — they're tiny but the real model is in ag_path
                    _is_ag_ref = len(_blob) < 500  # AG reference pkl is tiny
                    if not _is_ag_ref:
                        result["model_b64"] = _b64.b64encode(_blob).decode()
                        result["model_ext"] = _model_src.suffix or ".pkl"
                    RUNS[run_id]["result"] = result
                    print(f"[models] {'AG ref' if _is_ag_ref else f'Embedded {len(_blob)//1024}KB'} model ({_model_src.name}) for {run_id}", flush=True)
                except Exception as _mbe:
                    print(f"[models] model save failed: {_mbe}", flush=True)
                try:
                    shutil.copy2(_model_src, _MODELS_DIR / f"{run_id}{_model_src.suffix or '.pkl'}")
                except Exception:
                    pass
            else:
                print(f"[models] WARNING: no model found for {run_id} — files: {list(ws.rglob('*'))[:20] if ws.exists() else 'ws gone'}", flush=True)
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
        _kaggle_context_hint = ""
        _companion_profiles = {}
        if req.dataset_id:
            media = _MEDIA_DATASETS.get(req.dataset_id)
            if not media:
                return {"ok": False, "error": "Dataset not found. Please re-upload."}
            if media.get("type") == "tabular_dir":
                # Copy primary CSV to temp workspace for profiling
                primary_p = Path(media["primary"])
                csv_path = ws / primary_p.name
                shutil.copy2(primary_p, csv_path)
                data_path = str(csv_path)
                profile = profile_dataset(data_path)
                # Build Kaggle context hint from companion files
                all_file_paths = [Path(f) for f in media.get("all_files", [])]
                all_file_names = [p.name for p in all_file_paths]
                _test_f  = next((p for p in all_file_paths if "test" in p.name.lower() and p.exists()), None)
                _samp_f  = next((p for p in all_file_paths if ("sample" in p.name.lower() or "submission" in p.name.lower()) and p.exists()), None)
                # Quick-profile companion files (headers + row count only)
                _companion_profiles = {}
                for _fp in all_file_paths:
                    if _fp == primary_p or not _fp.exists():
                        continue
                    try:
                        import pandas as _pd
                        _cdf = _pd.read_csv(_fp, nrows=5)
                        with open(_fp) as _fh:
                            _nrows = sum(1 for _ in _fh) - 1
                        _role = "test" if "test" in _fp.name.lower() else "submission"
                        _companion_profiles[_fp.name] = {
                            "rows": _nrows, "cols": len(_cdf.columns),
                            "headers": list(_cdf.columns), "role": _role,
                        }
                    except Exception:
                        pass
                if _test_f or _samp_f:
                    _ctx_parts = [f"Workspace files: {', '.join(all_file_names)}."]
                    _ctx_parts.append(f"Train on {primary_p.name}.")
                    if _test_f:
                        _ctx_parts.append(f"Generate predictions for {_test_f.name} (no target column — unlabelled holdout).")
                    if _samp_f:
                        try:
                            import pandas as _pd
                            _samp_cols = list(_pd.read_csv(_samp_f, nrows=3).columns)
                            _ctx_parts.append(f"Output must match {_samp_f.name} format: columns={_samp_cols}. The last column is the TARGET to predict.")
                            _ctx_parts.append(f"Save final predictions as submission.csv.")
                        except Exception:
                            _ctx_parts.append(f"Save final predictions as submission.csv matching {_samp_f.name}.")
                    _kaggle_context_hint = " [COMPETITION CONTEXT: " + " ".join(_ctx_parts) + "]"
            else:
                data_path = media["path"]
                profile = profile_media_dataset(data_path)
            cleanup_ws = False  # don't delete the media dataset dir
        else:
            csv_path = ws / req.filename
            csv_path.write_text(req.csv, encoding="utf-8")
            data_path = str(csv_path)
            profile = profile_dataset(data_path)

        # Merge user hint with Kaggle context so the LLM sees both
        _effective_hint = ((req.hint or "") + _kaggle_context_hint).strip()

        resolved_api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        if not resolved_api_key:
            fallback = _fallback_discovery(profile, _effective_hint)
            fallback["ok"] = True
            fallback["provider_note"] = "Using smart fallback discovery. Add an API key for richer AI analysis."
            return fallback

        result = discover_user_need(data_path, user_hint=_effective_hint, previous_objective=req.previous_objective, api_key=resolved_api_key, provider=req.provider or "claude", model=req.model or None, companion_profiles=_companion_profiles or None)
        result["used_fallback"] = False
        return {"ok": True, **result}
    except Exception as e:
        import traceback as _tb
        tb = _tb.format_exc()
        print(f"[discover] ERROR: {e}\n{tb}", flush=True)
        msg = str(e).strip()
        if not msg:
            msg = f"{type(e).__name__}: {tb.splitlines()[-1] if tb else 'unknown error'}"
        if "Connection error" in msg:
            msg = "API connection error during discovery. Check internet/VPN/firewall/proxy."
        elif "Invalid API key" in msg or "authentication" in msg.lower() or "api_key" in msg.lower():
            msg = "Invalid API key. Please check your key in Settings."
        elif "ModuleNotFoundError" in msg or "ImportError" in msg:
            msg = f"Missing dependency: {msg}. Try re-deploying or contact support."
        return {"ok": False, "error": msg or "Unexpected server error during discovery", "used_fallback": False}
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

# ── DELETE RUN ─────────────────────────────────────────────────
@app.delete("/api/run/{run_id}")
def delete_run(run_id: str, request: Request):
    user = _get_user(_token_from_request(request))
    if not user:
        raise HTTPException(401, "Sign in to delete runs")
    user_id = user["id"]
    # Check ownership in memory
    if run_id in RUNS:
        if RUNS[run_id].get("owner_id") != user_id:
            raise HTTPException(403, "Access denied")
        del RUNS[run_id]
    # Delete from DB
    try:
        conn = DBConn()
        row = conn.execute("SELECT user_id FROM runs WHERE id=?", (run_id,)).fetchone()
        if row:
            if row["user_id"] != user_id:
                conn.close()
                raise HTTPException(403, "Access denied")
            conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
            conn.commit()
        conn.close()
    except HTTPException:
        raise
    except Exception:
        pass
    return {"ok": True}

# ── RENAME RUN ─────────────────────────────────────────────────
class RenameRunRequest(BaseModel):
    label: str

@app.patch("/api/run/{run_id}/label")
def rename_run(run_id: str, req: RenameRunRequest, request: Request):
    user = _get_user(_token_from_request(request))
    if not user:
        raise HTTPException(401, "Sign in to rename runs")
    user_id = user["id"]
    label = req.label.strip()[:80]
    if not label:
        raise HTTPException(400, "Label cannot be empty")
    # Update in memory
    if run_id in RUNS:
        if RUNS[run_id].get("owner_id") != user_id:
            raise HTTPException(403, "Access denied")
        RUNS[run_id]["label"] = label
    # Update in DB — store in hint field prefix (reuse existing column)
    try:
        conn = DBConn()
        row = conn.execute("SELECT user_id FROM runs WHERE id=?", (run_id,)).fetchone()
        if row:
            if row["user_id"] != user_id:
                conn.close()
                raise HTTPException(403, "Access denied")
            conn.execute("UPDATE runs SET filename=? WHERE id=?", (label, run_id))
            conn.commit()
        conn.close()
    except HTTPException:
        raise
    except Exception:
        pass
    return {"ok": True, "label": label}

# ── CLAIM ORPHANED RUNS (guest → auth user) ──────────────────────────────────
class ClaimRunsRequest(BaseModel):
    run_ids: list

@app.post("/api/runs/claim")
async def claim_runs(request: Request, body: ClaimRunsRequest):
    """Claim ownerless (guest) runs by assigning them to the authenticated user."""
    user = _get_user(_token_from_request(request))
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    run_ids = [str(r) for r in (body.run_ids or [])[:50]]
    if not run_ids:
        return {"claimed": 0}
    claimed = 0
    try:
        conn = DBConn()
        for rid in run_ids:
            conn.execute(
                "UPDATE runs SET user_id=? WHERE id=? AND user_id IS NULL",
                (user["id"], rid)
            )
            claimed += conn._cur.rowcount or 0
        conn.commit()
        conn.close()
    except Exception:
        pass
    # Also claim any in-memory runs that are ownerless and match these IDs
    for rid in run_ids:
        if rid in RUNS and RUNS[rid].get("owner_id") is None:
            RUNS[rid]["owner_id"] = user["id"]
    return {"claimed": claimed}

# ── LIST RUNS ──────────────────────────────────────────────────
@app.get("/api/runs")
def list_runs(request: Request):
    user = _get_user(_token_from_request(request))
    user_id = user["id"] if user else None
    out = []
    # In-memory runs — strict ownership:
    # auth users see ONLY their own; guests see ONLY ownerless runs
    for rid, run in sorted(RUNS.items(), key=lambda x: x[1].get("started", 0), reverse=True):
        owner = run.get("owner_id")
        if user_id:
            if owner != user_id:
                continue  # auth user — only their own runs
        else:
            if owner is not None:
                continue  # guest — only ownerless (guest) runs
        out.append({
            "id": rid,
            "status": run["status"],
            "started": run.get("started"),
            "filename": Path(run.get("csv", "")).name if run.get("csv") else "",
            "provider": run.get("provider", "claude"),
            "best_model": (run.get("result") or {}).get("best", {}).get("model"),
            "best_metric": (run.get("result") or {}).get("best", {}).get("metric_val"),
        })
    # Historical runs from DB — filtered by user_id at the query level
    in_memory_ids = set(RUNS.keys())
    for row in _load_run_history_from_db(user_id=user_id):
        if row["id"] not in in_memory_ids:
            out.append({
                "id": row["id"],
                "status": row["status"],
                "started": row["started"],
                "filename": row.get("filename", ""),
                "provider": row.get("provider", "claude"),
                "best_model": row.get("best_model"),
                "best_metric": row.get("best_metric_val"),
                "historical": True,
            })
    out.sort(key=lambda x: x.get("started") or 0, reverse=True)
    # For authenticated users, also surface unclaimed guest runs so they can claim them
    unclaimed = []
    if user_id:
        try:
            conn2 = DBConn()
            rows2 = conn2.execute(
                "SELECT * FROM runs WHERE user_id IS NULL AND id LIKE 'guest-%' ORDER BY started DESC LIMIT 20"
            ).fetchall()
            conn2.close()
            existing_ids = {r["id"] for r in out}
            for row in rows2:
                if row["id"] not in existing_ids:
                    unclaimed.append({
                        "id": row["id"],
                        "status": row["status"],
                        "started": row["started"],
                        "filename": row.get("filename", ""),
                        "provider": row.get("provider", ""),
                        "best_model": row.get("best_model"),
                        "unclaimed": True,
                    })
        except Exception:
            pass
    return {"runs": out[:100], "unclaimed": unclaimed}

@app.get("/api/stats")
def get_stats():
    """Dashboard stats -- total runs, models trained, best scores."""
    try:
        conn = DBConn()
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
def get_status(run_id: str, request: Request):
    requester = _get_user(_token_from_request(request))
    requester_id = requester["id"] if requester else None
    if run_id not in RUNS:
        # Try to reconstruct from DB
        try:
            conn = DBConn()
            row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            conn.close()
            if row:
                # Access control: auth users only see their own; guests only see ownerless
                try:
                    row_owner = row["user_id"]
                except (IndexError, KeyError):
                    row_owner = None
                if requester_id and row_owner and row_owner != requester_id:
                    raise HTTPException(403, "Access denied")
                if not requester_id and row_owner:
                    raise HTTPException(403, "Access denied")
                result = json.loads(row["result_json"]) if row["result_json"] else None
                elapsed = round((row["finished"] or time.time()) - (row["started"] or time.time()), 1)
                out = dict(
                    status=row["status"],
                    log_count=0,
                    elapsed=elapsed,
                    from_db=True,
                    hint=row["hint"],
                    filename=row["filename"],
                )
                if result:
                    out["best"]           = result.get("best")
                    out["objective"]      = result.get("objective")
                    out["history"]        = result.get("history")
                    out["report"]         = result.get("report")
                    out["deploy_path"]    = result.get("deploy_path")
                    out["diagnostics"]    = result.get("diagnostics")
                    out["executive_brief"]= result.get("executive_brief")
                    # Augment artifacts with workspace scan for any missing plot files
                    db_run = RUNS.get(run_id, {})
                    out["artifacts"]      = _augment_artifacts(db_run, result.get("artifacts"))
                    out["total_experiments"] = result.get("total_experiments", len(result.get("history", [])))
                    out["continuous_mode"]= result.get("continuous_mode", False)
                    out["token_usage"]    = result.get("token_usage", {})
                    # Code and logs persisted into result_json
                    if result.get("train_code"):
                        out["code"] = result["train_code"]
                    if result.get("run_logs"):
                        out["logs"] = result["run_logs"]
                elif row["best_model"]:
                    # No full result_json but we have summary fields
                    out["best"] = {
                        "model": row["best_model"],
                        "metric_name": row["best_metric_name"],
                        "metric_val": row["best_metric_val"],
                    }
                if row["error"]:
                    out["error"] = row["error"]
                return out
        except Exception:
            pass
        raise HTTPException(404, "Run not found")
    run = RUNS[run_id]
    # Access control for in-memory runs
    run_owner = run.get("owner_id")
    if requester_id and run_owner and run_owner != requester_id:
        raise HTTPException(403, "Access denied")
    if not requester_id and run_owner:
        raise HTTPException(403, "Access denied")
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
        out["artifacts"] = _augment_artifacts(run, result.get("artifacts"))
        out["total_experiments"] = result.get("total_experiments", len(result.get("history", [])))
        out["continuous_mode"] = result.get("continuous_mode", False)
        out["token_usage"] = result.get("token_usage", {})
        if result.get("train_code"):
            out["code"] = result["train_code"]
    # Also expose code from workspace file (for live runs before result is set)
    if "code" not in out:
        ws_path = Path(run.get("ws", ""))
        train_py = ws_path / "train.py"
        if train_py.exists():
            try:
                out["code"] = train_py.read_text()
            except Exception:
                pass
    # Expose logs for completed runs
    if run["status"] in ("done", "error") and result and result.get("run_logs"):
        out["logs"] = result["run_logs"]
    if run.get("error"):
        out["error"] = run["error"]
    return out


_KNOWN_PLOT_FILES = {
    "progress_png":              "progress.png",
    "experiment_timeline_png":   "experiment_timeline.png",
    "timeseries_png":            "timeseries.png",
    "correlation_png":           "correlation.png",
    "shap_png":                  "shap.png",
    "predictions_png":           "predictions.png",
    "residuals_png":             "residuals.png",
    "train_test_png":            "train_test.png",
    "model_comparison_png":      "model_comparison.png",
    "metrics_overview_png":      "metrics_overview.png",
    "data_overview_png":         "data_overview.png",
    "seasonality_png":           "seasonality.png",
    "feature_importance_png":    "feature_importance.png",
    "lag_correlation_png":       "lag_correlation.png",
}

def _augment_artifacts(run: dict, arts: dict) -> dict:
    """Scan the workspace for known plot files and add any that exist but aren't registered."""
    arts = dict(arts or {})
    ws = Path(run.get("ws", ""))
    if not ws.exists():
        return arts
    for key, fname in _KNOWN_PLOT_FILES.items():
        if not arts.get(key):
            p = ws / fname
            if p.exists():
                arts[key] = str(p)
    return arts


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

# ── DOWNLOAD MODEL ZIP (clean: model.pkl + train.py + requirements) ──
@app.get("/api/run/{run_id}/deploy")
def download_deploy(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "Run not found")
    result = RUNS[run_id].get("result")
    if not result:
        raise HTTPException(404, "Run not finished yet")
    ws = Path(RUNS[run_id].get("ws", ""))
    dp = Path(result["deploy_path"]) if result.get("deploy_path") else None

    zip_path = (dp.parent if dp else ws) / f"model_{run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # 1. model.pkl — prefer best_model.pkl (preserved at each new-best event)
        model_added = False
        search_dirs = [dp, ws] if dp and dp.exists() else [ws]
        for candidate_name in ["best_model.pkl", "model.pkl"]:
            for search_dir in search_dirs:
                candidate = search_dir / candidate_name
                if candidate.exists():
                    z.write(candidate, Path("model.pkl"))
                    model_added = True
                    break
            if model_added:
                break

        # 2. train.py (the winning experiment script)
        for src in ([dp / "train.py", ws / "train.py"] if dp and dp.exists() else [ws / "train.py"]):
            if src.exists():
                z.write(src, Path("train.py")); break

        # 3. requirements.txt
        for src in ([dp / "requirements.txt", ws / "requirements.txt"] if dp and dp.exists() else [ws / "requirements.txt"]):
            if src.exists():
                z.write(src, Path("requirements.txt")); break

        # 4. Minimal README
        best = result.get("best") or {}
        obj = result.get("objective") or {}
        readme = (
            f"# 19Labs Model — {best.get('model','ML Model')}\n\n"
            f"- Task: {obj.get('task','N/A')}\n"
            f"- Target: `{obj.get('target','N/A')}`\n"
            f"- {best.get('metric_name','metric').upper()}: **{best.get('metric_val','N/A')}**\n\n"
            f"## Load model\n```python\nimport joblib\nmodel = joblib.load('model.pkl')\npredictions = model.predict(X)\n```\n\n"
            f"## Retrain\n```bash\npip install -r requirements.txt\npython train.py\n```\n"
        )
        z.writestr("README.md", readme)

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
    allowed = {
        "program_md":              "program.md",
        "prepare_py":              "prepare.py",
        "analysis_ipynb":          "analysis.ipynb",
        "progress_png":            "progress.png",
        "train_test_png":          "train_test.png",
        "model_comparison_png":    "model_comparison.png",
        "metrics_overview_png":    "metrics_overview.png",
        "experiment_timeline_png": "experiment_timeline.png",
        "timeseries_png":            "timeseries.png",
        "correlation_png":          "correlation.png",
        "shap_png":                 "shap.png",
        "predictions_png":          "predictions.png",
        "residuals_png":            "residuals.png",
        "data_overview_png":        "data_overview.png",
        "seasonality_png":          "seasonality.png",
        "feature_importance_png":   "feature_importance.png",
        "lag_correlation_png":      "lag_correlation.png",
        "results_tsv":              "results.tsv",
        "train_py":                "train.py",
        "final_report_md":         "final_report.md",
    }
    is_exp_plot = artifact_name.startswith("exp_")
    if artifact_name not in allowed and not is_exp_plot:
        raise HTTPException(404, "Unknown artifact")

    # Resolve run + workspace — support both in-memory and post-restart DB lookups
    run = RUNS.get(run_id)
    ws_str = ""
    if run:
        result = run.get("result") or {}
        ws_str = run.get("ws", "")
        artifacts = _augment_artifacts(run, result.get("artifacts"))
    else:
        # Server restarted — load from DB
        try:
            conn = DBConn()
            row = conn.execute("SELECT result_json FROM runs WHERE id=?", (run_id,)).fetchone()
            conn.close()
        except Exception:
            row = None
        if not row:
            raise HTTPException(404, "Run not found")
        result = json.loads(row["result_json"]) if row["result_json"] else {}
        # Reconstruct workspace path: prefix was 19labs_{run_id}_  in tempdir
        import glob as _glob, tempfile as _tmp
        ws_candidates = _glob.glob(str(Path(_tmp.gettempdir()) / f"19labs_{run_id}_*"))
        if ws_candidates:
            ws_str = ws_candidates[0]
        else:
            # Try deriving from artifact paths stored in result
            for v in (result.get("artifacts") or {}).values():
                if v and Path(v).parent.exists():
                    ws_str = str(Path(v).parent)
                    break
        artifacts = _augment_artifacts({"ws": ws_str}, result.get("artifacts"))

    p = artifacts.get(artifact_name)
    if not p:
        # Final fallback: look directly in workspace by filename
        ws_path = Path(ws_str) if ws_str else None
        fname = allowed.get(artifact_name, "")
        if not fname and is_exp_plot:
            fname = artifact_name.replace("_png", ".png").replace("_py", ".py")
        if ws_path and ws_path.exists() and fname:
            candidate = ws_path / fname
            if candidate.exists():
                p = str(candidate)
        if not p:
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

    dl_filename = allowed.get(artifact_name) or artifact_name.replace("_png", ".png").replace("_py", ".py").replace("_md", ".md")
    return FileResponse(str(fp), filename=dl_filename, media_type=media)

# ── PREDICTIONS CSV (actuals vs predicted table) ───────────────
@app.get("/api/run/{run_id}/predictions")
def get_predictions(run_id: str, limit: int = 500):
    """Return actuals vs predicted CSV as JSON rows for the forecast table."""
    ws_str = None
    if run_id in RUNS:
        ws_str = RUNS[run_id].get("ws")
    else:
        try:
            from sqlalchemy.orm import Session as _S
            with _S(engine) as sess:
                result = sess.query(RunRecord).filter_by(run_id=run_id).first()
                if result:
                    ws_str = (result.result_json or {}).get("ws") if isinstance(result.result_json, dict) else None
                    if not ws_str:
                        import json as _j, glob as _g, tempfile as _tf
                        td = pathlib.Path(_tf.gettempdir())
                        matches = list(td.glob(f"19labs_{run_id}_*"))
                        if matches:
                            ws_str = str(matches[0])
        except Exception:
            pass
    if not ws_str:
        raise HTTPException(404, "Run not found")
    ws = Path(ws_str)
    # Try predictions.csv first, then any pred*.csv
    candidates = [ws / "predictions.csv"] + sorted(ws.glob("pred*.csv"), key=lambda f: -f.stat().st_mtime if f.exists() else 0)
    for csv_path in candidates:
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, nrows=limit)
                # Normalise column names
                col_map = {}
                for c in df.columns:
                    lc = c.lower().strip()
                    if lc in ("actual", "actuals", "y_true", "y_actual", "true"): col_map[c] = "actual"
                    elif lc in ("predicted", "prediction", "pred", "forecast", "y_pred"): col_map[c] = "predicted"
                    elif lc in ("date", "datetime", "time", "period", "timestamp"): col_map[c] = "date"
                df = df.rename(columns=col_map)
                # Compute error columns if actual+predicted present
                if "actual" in df.columns and "predicted" in df.columns:
                    df["error"] = (df["predicted"] - df["actual"]).round(4)
                    df["pct_error"] = ((df["error"] / df["actual"].replace(0, float("nan"))) * 100).round(2)
                rows = df.where(df.notna(), None).to_dict(orient="records")
                return {"ok": True, "columns": list(df.columns), "rows": rows, "total": len(rows)}
            except Exception as e:
                raise HTTPException(500, f"Could not parse predictions: {e}")
    raise HTTPException(404, "predictions.csv not found — run an experiment first")

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
        response = chat_with_data(req.message, req.context, api_key, req.provider, model=req.model or None)
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
    model: str = ""      # optional model override

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
    _init_client(api_key, req.provider or "claude", model=req.model or None)

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
    run = _get_run_or_404(run_id)
    result = run.get("result")
    if not result:
        raise HTTPException(400, "Run not finished yet")
    ws = Path(run.get("ws", "")) if run.get("ws") else None
    deploy_path = result.get("deploy_path")
    dp = Path(deploy_path) if deploy_path and Path(deploy_path).exists() else None
    # train_py from result_json (persisted) or workspace file
    train_py = result.get("train_code", "")
    if not train_py:
        for candidate_dir in [d for d in [dp, ws] if d and d.exists()]:
            candidate = candidate_dir / "train.py"
            if candidate.exists():
                train_py = candidate.read_text()
                break
    best = result.get("best") or {}
    obj = result.get("objective") or {}
    from engine import generate_inference_server
    generated = generate_inference_server(train_py, best, obj, api_key, provider)
    if "error" in generated and not generated.get("inference_server_py"):
        raise HTTPException(500, f"Generation failed: {generated['error']}")
    # Use stable dir for api output (ws may be gone after restart)
    _models_dir = Path(__file__).parent / "run_models"
    _models_dir.mkdir(exist_ok=True)
    api_dir = _models_dir / f"{run_id}_api"
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
    # Copy model — check stable run_models/ first, then workspace, then DB BLOB
    stable_model = next((_MODELS_DIR / f"{run_id}{ext}" for ext in [".pkl", ".joblib"] if (_MODELS_DIR / f"{run_id}{ext}").exists()), None)
    model_copied = False
    if stable_model:
        shutil.copy2(stable_model, api_dir / "model.pkl")
        model_copied = True
    if not model_copied:
        for src_name in ["best_model.pkl", "model.pkl"]:
            for search_dir in [d for d in [dp, ws] if d and d.exists()]:
                src = search_dir / src_name
                if src.exists() and src.is_file():
                    shutil.copy2(src, api_dir / "model.pkl")
                    model_copied = True
                    break
            if model_copied:
                break
    if not model_copied:
        # Last resort: load from DB BLOB
        try:
            _dbc = DBConn()
            _row = _dbc.execute("SELECT model_data, model_ext FROM run_models WHERE run_id=?", (run_id,)).fetchone()
            _dbc.close()
            if _row and _row["model_data"]:
                (api_dir / "model.pkl").write_bytes(bytes(_row["model_data"]))
                model_copied = True
        except Exception:
            pass
    zip_path = _models_dir / f"api_server_{run_id[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in api_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(api_dir))
    return FileResponse(str(zip_path), filename=f"19labs_api_{run_id[:8]}.zip", media_type="application/zip")

# ── BATCH PREDICTION ───────────────────────────────────────────
@app.post("/api/run/{run_id}/predict")
async def predict(run_id: str, req: PredictRequest):
    run = _get_run_or_404(run_id)
    result = run.get("result")
    if not result:
        raise HTTPException(400, "Run not finished yet")

    model_path = _find_model_path(run_id, run)
    if not model_path:
        _has_b64b = bool((result or {}).get("model_b64"))
        _ws_files_b = (result or {}).get("_ws_files", [])
        raise HTTPException(404, f"No model found. model_b64={'YES' if _has_b64b else 'NO'}, ws_files={_ws_files_b}")

    import joblib
    import pandas as pd
    import numpy as np

    model = None
    try:
        model = joblib.load(model_path)
    except Exception:
        pass
    if model is None:
        try:
            import xgboost as xgb; _b = xgb.Booster(); _b.load_model(str(model_path)); model = _b
        except Exception:
            pass
    if model is None:
        try:
            import catboost as cb; _c = cb.CatBoostClassifier(); _c.load_model(str(model_path)); model = _c
        except Exception:
            try:
                import catboost as cb; _c = cb.CatBoostRegressor(); _c.load_model(str(model_path)); model = _c
            except Exception as _le:
                raise HTTPException(500, f"Failed to load model: {_le}")

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


@app.post("/api/run/{run_id}/predict-file")
async def predict_file(run_id: str, file: UploadFile = File(...)):
    """Accept a CSV file upload, run predictions with the stored model, return submission.csv."""
    import joblib, pandas as pd, numpy as np, io as _io
    run = _get_run_or_404(run_id)
    result = run.get("result")
    if not result:
        raise HTTPException(400, "Run not finished yet")

    # Try AutoGluon first (most reliable — handles feature types automatically)
    _ag_path = _find_autogluon_path(run_id, run)
    contents = await file.read()
    try:
        df = pd.read_csv(_io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    obj = result.get("objective") or result.get("best") or {}
    target = obj.get("target", "")

    if _ag_path:
        try:
            _ag_pred = _load_autogluon_predictor(_ag_path)
            # AutoGluon handles everything — just drop target if present
            _df_pred = df.drop(columns=[c for c in [target] if c and c in df.columns])
            preds = _ag_pred.predict(_df_pred).values
            id_col = next((c for c in df.columns if c.lower() in ("id","customerid","customer_id","passengerid")), None)
            id_vals = df[id_col] if id_col else pd.RangeIndex(len(df))
            sub_target = target or "prediction"
            sub = pd.DataFrame({id_col or "id": id_vals, sub_target: preds})
            _buf = _io.BytesIO(); sub.to_csv(_buf, index=False); _buf.seek(0)
            return StreamingResponse(_buf, media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="submission_{run_id[:8]}.csv"'})
        except Exception as _age:
            print(f"[predict-file] AutoGluon predict failed: {_age} — falling back to pkl", flush=True)

    model_path = _find_model_path(run_id, run)
    if not model_path:
        _has_b64 = bool((result or {}).get("model_b64"))
        _ws_files = (result or {}).get("_ws_files", [])
        raise HTTPException(404, f"No model found. model_b64={'YES' if _has_b64 else 'NO'}, ws_files={_ws_files}")

    model = None
    load_err = None
    try:
        _loaded = joblib.load(model_path)
        # Handle AG reference dict saved in pkl
        if isinstance(_loaded, dict) and _loaded.get("type") == "autogluon":
            _ap2 = _loaded.get("ag_path", "")
            if _ap2 and Path(_ap2).exists():
                _ag_pred2 = _load_autogluon_predictor(_ap2)
                _df_pred2 = df.drop(columns=[c for c in [target] if c and c in df.columns])
                preds2 = _ag_pred2.predict(_df_pred2).values
                id_col2 = next((c for c in df.columns if c.lower() in ("id","customerid","customer_id","passengerid")), None)
                id_vals2 = df[id_col2] if id_col2 else pd.RangeIndex(len(df))
                sub2 = pd.DataFrame({id_col2 or "id": id_vals2, target or "prediction": preds2})
                _buf2 = _io.BytesIO(); sub2.to_csv(_buf2, index=False); _buf2.seek(0)
                return StreamingResponse(_buf2, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="submission_{run_id[:8]}.csv"'})
        model = _loaded
    except Exception as _e1:
        load_err = _e1
    if model is None:
        try:
            import xgboost as xgb
            _bst = xgb.Booster(); _bst.load_model(str(model_path)); model = _bst
        except Exception as _e2:
            load_err = _e2
    if model is None:
        try:
            import catboost as cb
            try: _cbt = cb.CatBoostClassifier(); _cbt.load_model(str(model_path)); model = _cbt
            except Exception: _cbt = cb.CatBoostRegressor(); _cbt.load_model(str(model_path)); model = _cbt
        except Exception as _e3:
            load_err = _e3
    if model is None:
        raise HTTPException(500, f"Failed to load model ({model_path.suffix}): {load_err}")

    # Keep ID column aside
    id_col = next((c for c in df.columns if c.lower() in ("id", "customerid", "customer_id", "passengerid")), None)
    id_vals = df[id_col] if id_col else pd.RangeIndex(len(df))

    # Drop target if present
    features = df.drop(columns=[c for c in [target] if c and c in df.columns])

    # Try prediction; fallback to label-encoded version
    preds = None
    try:
        preds = model.predict(features)
    except Exception:
        try:
            from sklearn.preprocessing import LabelEncoder
            fe = features.copy()
            for col in fe.select_dtypes(include=["object", "category"]).columns:
                le = LabelEncoder()
                fe[col] = le.fit_transform(fe[col].astype(str))
            fe = fe.fillna(0)
            preds = model.predict(fe)
        except Exception as e2:
            raise HTTPException(400, f"Prediction failed: {e2}")

    # Build submission dataframe
    sub_target = target or "prediction"
    sub = pd.DataFrame({id_col or "id": id_vals, sub_target: preds})

    csv_out = sub.to_csv(index=False)
    filename = "submission.csv"
    return StreamingResponse(
        _io.BytesIO(csv_out.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
            if len(data) > 500 * 1024 * 1024:
                raise HTTPException(413, "File too large (>500MB)")
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


# ── DATA CONNECTORS ────────────────────────────────────────────
class ConnectRequest(BaseModel):
    connector: str          # postgresql | mysql | sqlserver | snowflake | bigquery
                            # databricks | redshift | mongodb | s3 | rest
    credentials: dict       # connector-specific fields
    query: str = ""         # SQL query or empty → SELECT * FROM table LIMIT n
    table: str = ""         # table name (used when query is empty)
    limit: int = 50000      # max rows to fetch

@app.post("/api/connect")
async def connect_datasource(req: ConnectRequest):
    """Connect to an external data source and return CSV."""
    import io
    try:
        import pandas as pd
    except ImportError:
        return {"ok": False, "error": "pandas not installed"}

    c = req.connector.lower()
    creds = req.credentials
    limit = max(1, min(req.limit, 200_000))
    query = req.query.strip()
    table = req.table.strip()

    def df_to_response(df, source_name="data"):
        if df.empty:
            return {"ok": False, "error": "Query returned no rows"}
        if len(df) > limit:
            df = df.head(limit)
        csv_text = df.to_csv(index=False)
        fname = f"{source_name.replace(' ','_')}.csv"
        return {"ok": True, "csv": csv_text, "filename": fname,
                "rows": len(df), "cols": len(df.columns)}

    def build_sql(table, limit):
        if not table:
            return None
        return f'SELECT * FROM {table} LIMIT {limit}'

    # ── PostgreSQL / Redshift ──────────────────────────────────
    if c in ("postgresql", "postgres", "redshift"):
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=creds.get("host",""),
                port=int(creds.get("port", 5432)),
                dbname=creds.get("database",""),
                user=creds.get("username",""),
                password=creds.get("password",""),
                connect_timeout=15,
            )
            sql = query or build_sql(table, limit)
            if not sql:
                return {"ok": False, "error": "Provide a SQL query or table name"}
            df = pd.read_sql(sql, conn)
            conn.close()
            return df_to_response(df, creds.get("database","data"))
        except ImportError:
            return {"ok": False, "error": "psycopg2 not installed on server"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── MySQL / MariaDB ───────────────────────────────────────
    if c in ("mysql", "mariadb"):
        try:
            import pymysql
            conn = pymysql.connect(
                host=creds.get("host",""),
                port=int(creds.get("port", 3306)),
                db=creds.get("database",""),
                user=creds.get("username",""),
                password=creds.get("password",""),
                connect_timeout=15,
            )
            sql = query or build_sql(table, limit)
            if not sql:
                return {"ok": False, "error": "Provide a SQL query or table name"}
            df = pd.read_sql(sql, conn)
            conn.close()
            return df_to_response(df, creds.get("database","data"))
        except ImportError:
            return {"ok": False, "error": "pymysql not installed on server"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── SQL Server ────────────────────────────────────────────
    if c in ("sqlserver", "mssql"):
        try:
            import pymssql
            conn = pymssql.connect(
                server=creds.get("host",""),
                port=int(creds.get("port", 1433)),
                database=creds.get("database",""),
                user=creds.get("username",""),
                password=creds.get("password",""),
                login_timeout=15,
            )
            sql = query or build_sql(table, limit)
            if not sql:
                return {"ok": False, "error": "Provide a SQL query or table name"}
            df = pd.read_sql(sql, conn)
            conn.close()
            return df_to_response(df, creds.get("database","data"))
        except ImportError:
            return {"ok": False, "error": "pymssql not installed on server"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Snowflake ─────────────────────────────────────────────
    if c == "snowflake":
        try:
            import snowflake.connector
            conn = snowflake.connector.connect(
                account=creds.get("account",""),
                user=creds.get("username",""),
                password=creds.get("password",""),
                warehouse=creds.get("warehouse",""),
                database=creds.get("database",""),
                schema=creds.get("schema","PUBLIC"),
                login_timeout=20,
            )
            sql = query or build_sql(table, limit)
            if not sql:
                return {"ok": False, "error": "Provide a SQL query or table name"}
            df = pd.read_sql(sql, conn)
            conn.close()
            return df_to_response(df, creds.get("database","snowflake"))
        except ImportError:
            return {"ok": False, "error": "snowflake-connector-python not installed on server"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Databricks ────────────────────────────────────────────
    if c == "databricks":
        try:
            from databricks import sql as dbsql
            conn = dbsql.connect(
                server_hostname=creds.get("host",""),
                http_path=creds.get("http_path",""),
                access_token=creds.get("token",""),
            )
            sql = query or build_sql(table, limit)
            if not sql:
                return {"ok": False, "error": "Provide a SQL query or table name"}
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
            conn.close()
            df = pd.DataFrame(rows, columns=cols)
            return df_to_response(df, "databricks")
        except ImportError:
            return {"ok": False, "error": "databricks-sql-connector not installed on server"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── BigQuery ──────────────────────────────────────────────
    if c == "bigquery":
        try:
            from google.cloud import bigquery as bq
            import json as _json
            sa_json = creds.get("service_account_json","")
            if sa_json:
                import tempfile, os
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(sa_json); fname_tmp = f.name
                client = bq.Client.from_service_account_json(fname_tmp)
                os.unlink(fname_tmp)
            else:
                client = bq.Client(project=creds.get("project",""))
            sql = query or (f'SELECT * FROM `{table}` LIMIT {limit}' if table else None)
            if not sql:
                return {"ok": False, "error": "Provide a SQL query or table name"}
            df = client.query(sql).to_dataframe()
            return df_to_response(df, creds.get("project","bigquery"))
        except ImportError:
            return {"ok": False, "error": "google-cloud-bigquery not installed on server"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── MongoDB ───────────────────────────────────────────────
    if c == "mongodb":
        try:
            import pymongo
            client = pymongo.MongoClient(
                creds.get("uri", f"mongodb://{creds.get('host','localhost')}:{creds.get('port',27017)}/"),
                serverSelectionTimeoutMS=10000,
            )
            db = client[creds.get("database","")]
            coll = client[creds.get("database","")][creds.get("collection","")]
            docs = list(coll.find({}, {"_id": 0}).limit(limit))
            if not docs:
                return {"ok": False, "error": "Collection is empty or not found"}
            df = pd.DataFrame(docs)
            return df_to_response(df, creds.get("collection","mongodb"))
        except ImportError:
            return {"ok": False, "error": "pymongo not installed on server"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── AWS S3 ────────────────────────────────────────────────
    if c == "s3":
        try:
            import boto3
            s3 = boto3.client(
                "s3",
                aws_access_key_id=creds.get("access_key",""),
                aws_secret_access_key=creds.get("secret_key",""),
                region_name=creds.get("region","us-east-1"),
            )
            bucket = creds.get("bucket","")
            key = creds.get("key","")    # object path
            obj = s3.get_object(Bucket=bucket, Key=key)
            content = obj["Body"].read()
            ext = key.rsplit(".",1)[-1].lower()
            if ext in ("xlsx","xls"):
                df = pd.read_excel(io.BytesIO(content))
            elif ext == "json":
                df = pd.read_json(io.BytesIO(content))
            elif ext == "parquet":
                df = pd.read_parquet(io.BytesIO(content))
            else:
                df = pd.read_csv(io.BytesIO(content))
            return df_to_response(df, key.rsplit("/",1)[-1])
        except ImportError:
            return {"ok": False, "error": "boto3 not installed on server"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── REST / JSON API ───────────────────────────────────────
    if c == "rest":
        try:
            import urllib.request
            url = creds.get("url","")
            headers = {}
            if creds.get("token"):
                headers["Authorization"] = f"Bearer {creds['token']}"
            if creds.get("api_key"):
                headers[creds.get("api_key_header","X-API-Key")] = creds["api_key"]
            r = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(r, timeout=20) as resp:
                data = resp.read().decode("utf-8")
            import json as _j
            parsed = _j.loads(data)
            # Handle common shapes: list, {data:[...]}, {results:[...]}, {items:[...]}
            if isinstance(parsed, list):
                df = pd.DataFrame(parsed)
            elif isinstance(parsed, dict):
                for key in ("data","results","items","records","rows","value"):
                    if key in parsed and isinstance(parsed[key], list):
                        df = pd.DataFrame(parsed[key]); break
                else:
                    df = pd.json_normalize(parsed)
            else:
                return {"ok": False, "error": "Unexpected JSON shape"}
            return df_to_response(df, "api_data")
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": f"Unknown connector: {req.connector}"}


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
    run = _get_run_or_404(run_id)
    result = run.get("result")
    if not result:
        raise HTTPException(400, "Run not finished yet")

    model_path = _find_model_path(run_id, run)
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
