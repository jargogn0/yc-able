#!/usr/bin/env python3
"""
19Labs Autoresearch Engine
True loop: profile → infer → write → execute → REAL metrics → decide → rewrite → iterate
Auto-fix on failure. Updates objective.md every round. Packages best model for deployment.
"""

import os, sys, json, subprocess, tempfile, time, shutil, pathlib, re, ast, threading, math
import pandas as pd
from anthropic import Anthropic
try:
    from anthropic import AnthropicBedrock
except ImportError:
    AnthropicBedrock = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def detect_packages():
    import importlib
    candidates = [
        'sklearn', 'numpy', 'pandas', 'scipy', 'joblib', 'statsmodels',
        'xgboost', 'lightgbm', 'catboost', 'torch', 'tensorflow',
        'prophet', 'shap', 'optuna',
    ]
    available = []
    for pkg in candidates:
        try:
            importlib.import_module(pkg)
            available.append(pkg)
        except ImportError:
            pass
        except Exception:
            # Native runtime/ABI issues should mark package unavailable, not crash startup.
            pass
    return available

AVAILABLE_PKGS = detect_packages()
AUTORESEARCH_DIR = pathlib.Path(__file__).parent / "autoresearch-master"

MODULE_TO_PIP = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "sentence_transformers": "sentence-transformers",
    "transformers": "transformers",
    "datasets": "datasets",
    "sktime": "sktime",
    "neuralprophet": "neuralprophet",
    "prophet": "prophet",
    "hdbscan": "hdbscan",
    "umap": "umap-learn",
    "catboost": "catboost",
    "optuna": "optuna",
    "shap": "shap",
    "statsmodels": "statsmodels",
    "scipy": "scipy",
}

_installed_session: set = set()

# Import aliases that are not real pip package names
_NOT_PIP_PACKAGES = {"tf", "np", "pd", "plt", "sns", "sp", "sm"}

# Redirect import names → correct pip package names
_PACKAGE_REDIRECTS = {
    "cv2":                    "opencv-python-headless",
    "sklearn":                "scikit-learn",
    "sentence_transformers":  "sentence-transformers",
    "PIL":                    "Pillow",
    "bs4":                    "beautifulsoup4",
    "yaml":                   "pyyaml",
    "dotenv":                 "python-dotenv",
}

_INSTALL_TIMEOUT = 480  # 8 minutes — enough for any package including torch/tensorflow

def auto_install_packages(code: str, log=None) -> list:
    """Parse imports from generated code and pip-install any missing packages."""
    import importlib
    modules = detect_imported_modules(code)
    to_install = []
    for mod in modules:
        pip_name = MODULE_TO_PIP.get(mod, mod)
        if pip_name in _installed_session:
            continue
        # Skip standard library modules
        if mod in {"json", "os", "sys", "re", "time", "math", "random", "pathlib",
                   "collections", "itertools", "functools", "typing", "abc", "io",
                   "csv", "datetime", "warnings", "copy", "string", "struct",
                   "hashlib", "base64", "urllib", "http", "threading", "subprocess",
                   "gc", "traceback", "inspect", "contextlib", "dataclasses",
                   "enum", "logging", "argparse", "pickle", "shelve", "sqlite3"}:
            continue
        # Skip bare import aliases (not real packages)
        if pip_name in _NOT_PIP_PACKAGES:
            continue
        # Redirect to correct pip name
        pip_name = _PACKAGE_REDIRECTS.get(pip_name, pip_name)
        try:
            importlib.import_module(mod)
        except ImportError:
            to_install.append(pip_name)
        except Exception:
            pass

    installed = []
    for pkg in to_install:
        try:
            if log:
                log.engine(f"Auto-installing {pkg}...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "-q", "--no-warn-script-location"],
                timeout=_INSTALL_TIMEOUT, capture_output=True, text=True
            )
            if result.returncode == 0:
                _installed_session.add(pkg)
                installed.append(pkg)
                if log:
                    log.engine(f"Installed {pkg} ✓")
            else:
                if log:
                    log.engine(f"Could not install {pkg}: {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            if log:
                log.engine(f"Install timed out for {pkg} — skipping")
        except Exception as e:
            if log:
                log.engine(f"Install skipped ({pkg}): {e}")
    return installed

MIN_REQUIREMENTS = {
    "anthropic": "anthropic>=0.44",
    "catboost": "catboost>=1.2",
    "fastapi": "fastapi>=0.115",
    "joblib": "joblib>=1.3",
    "lightgbm": "lightgbm>=4.0",
    "numpy": "numpy>=1.26",
    "optuna": "optuna>=3.0",
    "pandas": "pandas>=2.2",
    "prophet": "prophet>=1.1",
    "pydantic": "pydantic>=2.9",
    "python-dotenv": "python-dotenv>=1.0",
    "scikit-learn": "scikit-learn>=1.3",
    "scipy": "scipy>=1.10",
    "shap": "shap>=0.45",
    "statsmodels": "statsmodels>=0.14",
    "tensorflow": "tensorflow>=2.15",
    "torch": "torch>=2.2",
    "uvicorn": "uvicorn>=0.32",
    "xgboost": "xgboost>=2.0",
}

def detect_imported_modules(code):
    modules = set()
    try:
        tree = ast.parse(code)
    except Exception:
        return modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules

def build_requirements_from_code(*codes, extra_modules=None):
    modules = set(extra_modules or [])
    for code in codes:
        if code:
            modules.update(detect_imported_modules(code))
    # Core tabular stack defaults for robustness.
    modules.update({"pandas", "numpy", "joblib"})

    pip_pkgs = set()
    for mod in modules:
        pip_pkgs.add(MODULE_TO_PIP.get(mod, mod))

    lines = []
    for pkg in sorted(pip_pkgs):
        if pkg in MIN_REQUIREMENTS:
            lines.append(MIN_REQUIREMENTS[pkg])
    return "\n".join(lines) + "\n"

def write_workspace_requirements(ws, train_py_code):
    ws = pathlib.Path(ws)
    req_text = build_requirements_from_code(train_py_code)
    (ws / "requirements.txt").write_text(req_text)
    return req_text

# ── CONFIG ─────────────────────────────────────────────────────
CLAUDE_MODEL  = "claude-sonnet-4-6"
OPENAI_MODEL  = "gpt-4o"
# Active model overrides (set by _init_client when caller passes a model)
_active_claude_model  = CLAUDE_MODEL
_active_openai_model  = OPENAI_MODEL
# Bedrock model ID — can be overridden via BEDROCK_MODEL env var.
# Default is claude-3-5-sonnet which is widely available across regions.
# Cross-region inference prefix format: us.anthropic.claude-... (for us-east-1/us-west-2)
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL", "anthropic.claude-sonnet-4-6")
EXEC_TIMEOUT = 180          # hard wall-clock kill per experiment
TIME_BUDGET  = 120          # target training budget (seconds) injected into scripts
STAGNATION_LIMIT = 3
GOOD_ENOUGH  = {"r2": 0.90, "auc": 0.92, "f1": 0.88, "accuracy": 0.90, "mape": 0.10, "rmse": 5000, "mae": 3000, "nse": 0.85}
SECONDARY_METRICS = ["r2", "mape", "mae", "rmse", "nse"]  # always request these alongside primary
MAX_CONTINUOUS_EXPERIMENTS = 200   # hard cap for continuous mode safety

RELIABILITY_PROFILES = {
    "demo_safe": {
        "budget_cap": 6,
        "stagnation_limit": 4,
        "continuous_cap": 20,
        "policy": (
            "Prioritize robust, deterministic baselines. Use stable sklearn models and simple feature engineering. "
            "Avoid fragile complexity and path-sensitive code."
        ),
    },
    "balanced": {
        "budget_cap": 12,
        "stagnation_limit": 5,
        "continuous_cap": 100,
        "policy": (
            "Balance reliability and performance. Start from strong baseline, then iterate with controlled complexity."
        ),
    },
    "aggressive": {
        "budget_cap": 20,
        "stagnation_limit": 7,
        "continuous_cap": 200,
        "policy": (
            "Push for best metric with broader search and higher complexity while preserving correctness and reproducibility."
        ),
    },
}

_client = None
_provider = "claude"

# ── LOGGER ─────────────────────────────────────────────────────
class Logger:
    def __init__(self, ws, cb=None):
        self.ws = pathlib.Path(ws)
        self.cb = cb
        self.log_path = self.ws / "research.log"
    def _w(self, tag, msg):
        line = f"[{time.strftime('%H:%M:%S')}][{tag}] {msg}"
        with open(self.log_path, "a") as f: f.write(line + "\n")
        if self.cb: self.cb(tag.lower(), msg)
        print(line, flush=True)
    def engine(self, m): self._w("ENGINE", m)
    def claude(self, m): self._w("CLAUDE", m)
    def result(self, m): self._w("RESULT", m)
    def err(self,    m): self._w("ERROR",  m)
    def sys(self,    m): self._w("SYS",    m)

# ── GIT-BASED STATE MANAGEMENT (Karpathy protocol) ────────────
def _git(ws, *args, check=True):
    """Run a git command in the workspace. Returns stdout."""
    r = subprocess.run(
        ["git"] + list(args),
        cwd=str(ws), capture_output=True, text=True, timeout=30,
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr[:300]}")
    return r.stdout.strip()

def git_init_workspace(ws, run_tag):
    """Initialize a git repo in the workspace for experiment tracking."""
    ws = pathlib.Path(ws)
    try:
        _git(ws, "init")
        _git(ws, "checkout", "-b", f"autoresearch/{run_tag}")
        gitignore = ws / ".gitignore"
        gitignore.write_text("__pycache__/\n*.pyc\nmodel.pkl\nbest_model.pkl\nrun.log\n*.zip\ndeploy/\n")
        _git(ws, "add", "-A")
        _git(ws, "commit", "-m", "init: workspace setup", "--allow-empty")
        return True
    except Exception:
        return False

def git_commit_experiment(ws, exp_num, description):
    """Commit the current state of train.py after an experiment."""
    ws = pathlib.Path(ws)
    try:
        _git(ws, "add", "-A")
        _git(ws, "commit", "-m", f"exp {exp_num:02d}: {description[:120]}", "--allow-empty")
        return _git(ws, "rev-parse", "--short", "HEAD")
    except Exception:
        return ""

def git_discard_uncommitted(ws):
    """Restore all tracked files to HEAD state (discard uncommitted changes)."""
    ws = pathlib.Path(ws)
    try:
        r1 = subprocess.run(["git", "checkout", "--", "."], cwd=str(ws),
                            capture_output=True, timeout=15)
        r2 = subprocess.run(["git", "clean", "-fd"], cwd=str(ws),
                            capture_output=True, timeout=15)
        return r1.returncode == 0 and r2.returncode == 0
    except Exception:
        return False

def git_get_commit_hash(ws):
    """Get current short commit hash."""
    try:
        return _git(pathlib.Path(ws), "rev-parse", "--short", "HEAD")
    except Exception:
        return ""

# ── LIVE NARRATION ─────────────────────────────────────────────
def narrate(cb, event_type, **kwargs):
    """Generate intelligent real-time narration for the chat during experiments."""
    messages = {
        "profiling_start": "Scanning your dataset... looking at distributions, correlations, missing values, and data types.",
        "profiling_done": lambda: f"Found {kwargs.get('rows', '?'):,} rows × {kwargs.get('cols', '?')} columns. {kwargs.get('signals', '')}",
        "domain_analysis": "Analyzing what domain this data belongs to and what modeling strategy fits best...",
        "domain_done": lambda: f"Domain identified: {kwargs.get('domain', 'General')}. Strategy: {kwargs.get('strategy', 'standard ML pipeline')}",
        "writing_plan": "Writing the research plan (program.md) — this is the blueprint for all experiments.",
        "writing_code": lambda: f"Writing experiment #{kwargs.get('num', 1)} — {kwargs.get('approach', 'generating training code')}...",
        "installing": lambda: f"Installing {kwargs.get('pkg', 'packages')}... (auto-detected from imports)",
        "executing": lambda: f"Running experiment #{kwargs.get('num', 1)}... {kwargs.get('model', '')} training in progress.",
        "metrics_parsed": lambda: _narrate_metrics(kwargs),
        "keep_decision": lambda: f"KEEP — {kwargs.get('model', 'model')} achieved {kwargs.get('metric', 'metric')}={kwargs.get('val', 0):.4f}. {kwargs.get('reason', 'New best score.')}",
        "discard_decision": lambda: f"DISCARD — {kwargs.get('reason', 'Score did not improve.')} Trying a different approach next.",
        "crash_recovery": lambda: f"Experiment crashed ({kwargs.get('reason', 'error')}). Auto-repairing and retrying...",
        "hard_reset": "3 crashes in a row — hard resetting to last known good checkpoint via git.",
        "stagnation": "Stagnation detected — pivoting to a completely different modeling strategy.",
        "good_enough": lambda: f"Hit the quality threshold! {kwargs.get('metric', 'metric')}={kwargs.get('val', 0):.4f} is excellent for this domain.",
        "final_report": "Generating final report with findings, methodology, and deployment package...",
        "overfitting_warning": lambda: f"Overfitting detected on #{kwargs.get('num', '?')}: train={kwargs.get('train', 0):.4f} vs test={kwargs.get('test', 0):.4f}. Next experiment will add regularization.",
        "improvement": lambda: f"Improvement! {kwargs.get('metric', 'metric')} went from {kwargs.get('prev', 0):.4f} → {kwargs.get('curr', 0):.4f} ({kwargs.get('pct', 0):+.1f}%)",
        "predictions_ready": lambda: f"📊 {kwargs.get('summary', 'Forecast table ready')} — switch to the Results tab to see actual vs predicted values.",
    }

    template = messages.get(event_type)
    if not template:
        return
    msg = template() if callable(template) else template
    if cb:
        cb("narrate", msg)

def _narrate_metrics(kwargs):
    """Build a rich narration from experiment metrics."""
    model = kwargs.get('model', 'Model')
    metrics = kwargs.get('metrics', {})
    primary = kwargs.get('primary_metric', 'metric')
    val = kwargs.get('val')

    parts = [f"{model} finished"]
    if val is not None:
        parts.append(f"— {primary}={val:.4f}")

    # Detect overfitting
    train_key = f"train_{primary}"
    test_key = f"test_{primary}"
    if train_key in metrics and test_key in metrics:
        train_v = metrics[train_key]
        test_v = metrics[test_key]
        ratio = abs(train_v - test_v) / max(abs(test_v), 1e-10)
        if ratio > 0.3:
            parts.append(f"(overfitting: train={train_v:.4f}, test={test_v:.4f})")
        else:
            parts.append(f"(good generalization: train={train_v:.4f} ≈ test={test_v:.4f})")

    return " ".join(parts)

# ── PROFILE ────────────────────────────────────────────────────
def _read_csv_smart(csv_path):
    """Try multiple delimiters and encodings to load a CSV correctly."""
    encodings = ["utf-8", "latin-1", "utf-8-sig"]
    separators = [",", ";", "\t", "|"]
    last_err = None
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(csv_path, encoding=enc, sep=sep)
                # Accept if we got more than 1 column OR it's the first pass (comma/utf-8)
                if len(df.columns) > 1:
                    return df, sep, enc
                if sep == "," and enc == "utf-8":
                    last_df = df  # keep as fallback
            except Exception as e:
                last_err = e
    # Return best single-column result if nothing better found
    try:
        return pd.read_csv(csv_path), ",", "utf-8"
    except Exception as e:
        raise RuntimeError(f"Cannot read CSV: {last_err or e}") from (last_err or e)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".wma"}

def profile_media_dataset(data_dir):
    """Profile a directory of images or audio files. Returns a profile dict compatible with the CSV profile format."""
    data_dir = pathlib.Path(data_dir)
    all_files = [f for f in data_dir.rglob("*") if f.is_file() and not f.name.startswith(".")]
    image_files = [f for f in all_files if f.suffix.lower() in _IMAGE_EXTS]
    audio_files = [f for f in all_files if f.suffix.lower() in _AUDIO_EXTS]
    media_files = image_files if image_files else audio_files
    media_type = "image" if image_files else ("audio" if audio_files else "file")

    classes = {}
    rows_data = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir():
            cls_files = [f for f in item.rglob("*") if f.is_file() and f.suffix.lower() in (_IMAGE_EXTS if image_files else _AUDIO_EXTS)]
            if cls_files:
                classes[item.name] = len(cls_files)
                for f in cls_files:
                    rows_data.append({"filepath": str(f), "label": item.name})

    if not rows_data:
        for f in media_files:
            rows_data.append({"filepath": str(f), "label": ""})

    task_type = f"{media_type}_classification" if classes else media_type
    signals = [task_type, f"{len(media_files)}_files"]
    if classes:
        signals.append(f"{len(classes)}_classes")
    if not media_files:
        signals.append("no_media_files_found")

    # Build columns list in the same format as profile_dataset for compatibility
    # with analyze_domain, infer_objective, write_train_py, etc.
    columns_list = [
        {"name": "filepath", "type": "text", "unique": len(rows_data), "null_pct": 0,
         "avg_len": 50, "text_sample": rows_data[0]["filepath"] if rows_data else ""},
        {"name": "label", "type": "categorical", "unique": len(classes),
         "null_pct": 0, "top_values": list(classes.keys())[:5], "majority_pct": 0},
    ]

    return {
        "path": str(data_dir),
        "media_type": media_type,
        "task_type": task_type,
        "rows": len(rows_data),
        "cols": 2,
        "headers": ["filepath", "label"],
        "columns": columns_list,
        "numeric": [],
        "categorical": ["label"] if classes else [],
        "text": ["filepath"],
        "datetime": [],
        "target_candidates": ["label"],
        "class_balance": {k: v for k, v in classes.items()},
        "top_correlations": [],
        "signals": signals,
        "classes": classes,
        "num_classes": len(classes),
        "total_files": len(media_files),
        "sample_files": [str(f.relative_to(data_dir)) for f in media_files[:6]],
        "missing": {},
        "outlier_cols": [],
        "detected_sep": None,
        "is_media": True,
    }


def profile_dataset(csv_path):
    df, detected_sep, detected_enc = _read_csv_smart(csv_path)
    if df.empty:
        raise RuntimeError("CSV file is empty (0 rows). Please upload a dataset with data.")
    if len(df.columns) == 1:
        # Single column — likely wrong delimiter; add a warning to the profile
        _single_col_warning = f"WARNING: Only 1 column detected (sep={repr(detected_sep)}). Data may be improperly delimited."
    else:
        _single_col_warning = None
    cols = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            cols.append(dict(name=col, type="empty", unique=0, nulls=len(df), sample="[]"))
            continue

        nr = pd.to_numeric(s, errors='coerce').notna().mean()
        is_date = False
        if nr < 0.7:
            try:
                pd.to_datetime(s.iloc[:5])
                if any("-" in str(v) or "/" in str(v) for v in s.iloc[:3]):
                    is_date = True
            except: pass

        str_s = s.astype(str)
        avg_len = str_s.str.len().mean()
        is_long_text = avg_len > 40 and nr < 0.3
        is_path = avg_len > 5 and str_s.str.contains(
            r'\.(jpg|jpeg|png|gif|bmp|tiff|webp|csv|json|txt)$', case=False, regex=True).mean() > 0.5

        if is_date:            typ = "datetime"
        elif nr > 0.8:         typ = "numeric"
        elif is_path:          typ = "filepath"
        elif is_long_text:     typ = "text"
        elif s.nunique() / max(len(s), 1) < 0.05: typ = "categorical"
        else:                  typ = "high_cardinality"

        col_info = dict(
            name=col, type=typ,
            unique=int(s.nunique()),
            nulls=int(df[col].isna().sum()),
            null_pct=round(df[col].isna().mean() * 100, 1),
            sample=str(list(s.dropna().iloc[:3].values)),
        )

        if typ == "numeric":
            num = pd.to_numeric(s, errors='coerce').dropna()
            def _sf(v, digits=4):
                """Safe float — converts NaN/inf to None for JSON compliance."""
                try:
                    f = float(v)
                    if f != f or f == float('inf') or f == float('-inf'):
                        return None
                    return round(f, digits)
                except Exception:
                    return None
            col_info.update(
                mean=_sf(num.mean()),
                std=_sf(num.std()),
                min=_sf(num.min()),
                max=_sf(num.max()),
                p25=_sf(num.quantile(0.25)),
                p50=_sf(num.quantile(0.50)),
                p75=_sf(num.quantile(0.75)),
                skew=_sf(num.skew(), 3),
                # Serial correlation — high value suggests time-series ordering
                serial_corr=_sf(num.autocorr(lag=1) if len(num) > 2 else 0.0, 3),
            )
        elif typ in ("categorical", "high_cardinality"):
            vc = s.value_counts()
            col_info.update(
                top_values=vc.head(5).to_dict(),
                # Class balance ratio — useful for imbalanced detection
                majority_pct=round(float(vc.iloc[0] / len(s) * 100), 1) if len(vc) > 0 else 100.0,
            )
        elif typ == "text":
            col_info.update(
                avg_len=round(avg_len, 1),
                text_sample=str(s.iloc[0])[:200] if len(s) > 0 else "",
            )

        cols.append(col_info)

    text_cols     = [c["name"] for c in cols if c["type"] == "text"]
    datetime_cols = [c["name"] for c in cols if c["type"] == "datetime"]
    filepath_cols = [c["name"] for c in cols if c["type"] == "filepath"]
    numeric_cols  = [c["name"] for c in cols if c["type"] == "numeric"]
    cat_cols      = [c["name"] for c in cols if c["type"] in ("categorical", "high_cardinality")]

    # Class balance on low-cardinality columns (potential targets)
    class_balance = {}
    for col in df.columns:
        s = df[col].dropna()
        if 1 < s.nunique() <= 20:
            vc = s.value_counts(normalize=True).round(3)
            class_balance[col] = vc.to_dict()

    # Top pairwise correlations (numeric only, exclude ID-like columns)
    top_correlations = []
    num_df = df[numeric_cols].copy() if numeric_cols else pd.DataFrame()
    # Drop likely ID columns (monotonically increasing int, near-unique)
    id_like = [c for c in num_df.columns if num_df[c].nunique() / max(len(num_df), 1) > 0.95]
    num_df = num_df.drop(columns=id_like, errors='ignore')
    if len(num_df.columns) >= 2:
        try:
            corr = num_df.corr().abs()
            pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                     .stack().sort_values(ascending=False))
            for (c1, c2), v in pairs.head(8).items():
                fv = float(v)
                if fv == fv and fv != float('inf'):  # skip NaN/inf
                    top_correlations.append({"cols": [c1, c2], "corr": round(fv, 3)})
        except Exception:
            pass

    # Dataset-level signals (concrete, actionable)
    signals = []
    if text_cols:
        samples = {c: df[c].dropna().iloc[0][:120] if len(df[c].dropna()) > 0 else "" for c in text_cols[:2]}
        signals.append(f"TEXT_COLUMNS {text_cols}: real samples={samples} → NLP task likely")
    if datetime_cols and numeric_cols:
        high_sc = [c["name"] for c in cols if c.get("serial_corr", 0) and abs(c["serial_corr"]) > 0.5]
        signals.append(f"DATETIME+NUMERIC detected → time-series/forecasting likely; high serial-corr cols: {high_sc or 'check manually'}")
    if filepath_cols:
        signals.append(f"FILEPATH_COLUMNS {filepath_cols} → image/file-based ML likely")
    # Imbalance signal
    for col, balance in class_balance.items():
        vals = list(balance.values())
        if vals and max(vals) > 0.85:
            signals.append(f"IMBALANCED_TARGET candidate '{col}': {balance} — majority {max(vals)*100:.0f}% → use stratified CV, class weights, AUC/F1 not accuracy")
    if not signals:
        signals.append("TABULAR dataset — standard supervised learning")

    if _single_col_warning:
        signals.insert(0, _single_col_warning)

    _n_rows = len(df)
    def _is_id_like(c):
        if c["type"] != "numeric":
            return False
        name_lo = c["name"].lower()
        if name_lo in ("id", "index", "row_id", "rowid", "record_id") or name_lo.endswith("_id") or name_lo.endswith("id"):
            return True
        if _n_rows > 0 and c["unique"] / _n_rows > 0.95:
            return True
        return False

    def _build_target_candidates(cols, n_rows):
        # Well-known target column names — ranked highest
        _TARGET_NAMES = {
            "churn", "target", "label", "class", "y", "outcome", "default",
            "fraud", "survived", "response", "converted", "clicked", "purchased",
            "defaulted", "attrition", "cancelled", "retained", "approved",
            "spam", "toxic", "sentiment", "result", "status", "success",
        }
        candidates = []
        for c in cols:
            name_lo = c["name"].lower()
            is_id = _is_id_like(c)
            is_known_target = name_lo in _TARGET_NAMES
            # Include: low-cardinality categoricals (2–50 unique) — classification targets
            is_cat_target = c["type"] in ("categorical", "high_cardinality") and 2 <= c["unique"] <= 50
            # Include: numeric with enough signal, not ID-like
            is_num_target = c["type"] == "numeric" and c["unique"] > 10 and not is_id
            if is_known_target or is_cat_target or is_num_target:
                # Score: known target names first, then low-cardinality cats, then numeric
                score = 0 if is_known_target else (1 if is_cat_target else 2)
                candidates.append((score, c["name"]))
        candidates.sort(key=lambda x: x[0])
        return [name for _, name in candidates[:5]]

    return dict(
        path=str(csv_path), rows=len(df), cols=len(df.columns),
        headers=list(df.columns), columns=cols,
        numeric=numeric_cols, categorical=cat_cols,
        datetime=datetime_cols, text=text_cols, filepath=filepath_cols,
        signals=signals,
        class_balance=class_balance,
        top_correlations=top_correlations,
        detected_sep=detected_sep,
        target_candidates=_build_target_candidates(cols, _n_rows),
    )


# ── DOMAIN INTELLIGENCE ────────────────────────────────────────
def analyze_domain(profile, hint=""):
    """
    Deep domain analysis — reasons like a senior data scientist.
    Understands the industry, problem type, data quality issues, and
    produces a concrete expert modeling strategy before any training starts.
    """
    # Build a rich, concise view Claude can reason over
    col_lines = []
    for c in profile["columns"]:
        line = f"  {c['name']} [{c['type']}] unique={c['unique']} nulls={c['null_pct']}%"
        if c["type"] == "numeric":
            line += f" | mean={c.get('mean')} std={c.get('std')} range=[{c.get('min')},{c.get('max')}] skew={c.get('skew')} serial_corr={c.get('serial_corr')}"
        elif c["type"] in ("categorical", "high_cardinality"):
            line += f" | top={c.get('top_values')} majority={c.get('majority_pct')}%"
        elif c["type"] == "text":
            line += f" | avg_len={c.get('avg_len')} sample: \"{c.get('text_sample','')[:100]}\""
        col_lines.append(line)
    col_detail = "\n".join(col_lines)

    balance_txt = "\n".join(
        f"  {col}: {dist}" for col, dist in list(profile.get("class_balance", {}).items())[:8]
    ) or "  (none detected)"

    corr_txt = "\n".join(
        f"  {p['cols'][0]} ↔ {p['cols'][1]}: {p['corr']}"
        for p in profile.get("top_correlations", [])[:6]
    ) or "  (none)"

    signals_txt = "\n".join(f"  {s}" for s in profile.get("signals", []))

    resp = ask(
        "You are a world-class senior data scientist and ML engineer. "
        "You read datasets the way a doctor reads an X-ray — instantly seeing what matters. "
        "Your job is to produce an expert analysis that will drive all modeling decisions.",
        f"""Analyze this dataset deeply. Think step by step like a senior ML engineer at a top tech company.

═══════════════════════════════════════
DATASET: {profile['rows']:,} rows × {profile['cols']} cols
FILE: {pathlib.Path(profile['path']).name}
═══════════════════════════════════════

COLUMNS (name, type, stats):
{col_detail}

CLASS BALANCE (low-cardinality columns):
{balance_txt}

TOP CORRELATIONS:
{corr_txt}

DATASET SIGNALS:
{signals_txt}

USER HINT: {hint or "(none — infer everything from data)"}

═══════════════════════════════════════
PRODUCE A STRUCTURED EXPERT ANALYSIS:
═══════════════════════════════════════

INDUSTRY: <specific industry/domain e.g. "Healthcare — ICU readmission prediction", "E-commerce — customer churn", "Finance — credit default", "NLP — product sentiment">
PROBLEM_TYPE: <precise ML task e.g. "Binary classification with severe class imbalance (8% positive)", "Multivariate time-series forecasting (daily granularity)", "NLP sentiment regression on product reviews">
TARGET_COLUMN: <exact column name, with reasoning>
TARGET_METRIC: <best metric AND why e.g. "ROC-AUC: class imbalance makes accuracy useless", "RMSE after log-transform: revenue is log-normal", "Macro-F1: multi-class with imbalance">
DATA_QUALITY: <specific issues: missing patterns, outliers, leakage risks, skew, cardinality problems — be concrete>
KEY_INSIGHTS: <3-5 bullet points — what a senior DS would notice immediately e.g. "• Serial corr=0.91 on 'sales' — strong AR signal", "• 'customer_id' leaks target — must exclude", "• 'review_text' avg 180 chars — BERT fine-tuning likely optimal">
MODELING_STRATEGY: <concrete 3-tier strategy>
  BASELINE: <specific model + specific reason, e.g. "LightGBM with scale_pos_weight=12 for imbalance">
  BETTER: <specific upgrade, e.g. "Optuna-tuned XGBoost + SMOTE oversampling + 5-fold stratified CV">
  ADVANCED: <specific advanced approach, e.g. "Stacking: LGBM + CatBoost + LR meta-learner, calibrated with Platt scaling">
CRITICAL_WARNINGS: <what will go wrong if ignored, e.g. "Random splits on time-series = data leakage. Must use walk-forward validation.", "Log-transform revenue or RMSE will be dominated by outliers.">
EXPECTED_PERFORMANCE: <realistic range e.g. "AUC 0.75-0.85 achievable; >0.90 would suggest leakage">""",
        1800
    )
    return resp


def _parse_domain_field(resp, key):
    m = re.search(rf"^{key}:\s*(.+?)(?=\n[A-Z_]+:|$)", resp, re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""

def _safe_float(s, default=0.0):
    try:
        return float(s)
    except (ValueError, TypeError):
        return default

# ── TOKEN TRACKING ─────────────────────────────────────────────
_token_usage = {"input": 0, "output": 0, "calls": 0}

def get_token_usage():
    return dict(_token_usage)

def reset_token_usage():
    _token_usage["input"] = 0
    _token_usage["output"] = 0
    _token_usage["calls"] = 0

# ── LLM (multi-provider) ───────────────────────────────────────
def _init_client(api_key, provider="claude", model=None):
    """Initialise the LLM client.

    For Bedrock, api_key is a JSON string:
      {"access_key": "AKIA...", "secret_key": "...", "region": "us-east-1"}
    model: optional override for the model to use for this session.
    """
    global _client, _provider, _active_claude_model, _active_openai_model, BEDROCK_MODEL
    _provider = (provider or "claude").lower()
    if model:
        if _provider == "openai":
            _active_openai_model = model
        elif _provider == "bedrock":
            BEDROCK_MODEL = model
        else:
            _active_claude_model = model
    else:
        # Reset to defaults when no model override given
        _active_claude_model = CLAUDE_MODEL
        _active_openai_model = OPENAI_MODEL
    if _provider == "openai":
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        _client = OpenAI(api_key=api_key)
    elif _provider == "bedrock":
        if AnthropicBedrock is None:
            raise RuntimeError("anthropic package too old for Bedrock. Run: pip install 'anthropic>=0.31'")
        try:
            creds = json.loads(api_key) if isinstance(api_key, str) and api_key.startswith("{") else {}
        except Exception:
            creds = {}
        ak = creds.get("access_key") or os.environ.get("AWS_ACCESS_KEY_ID", "")
        sk = creds.get("secret_key") or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        rg = creds.get("region") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        if not ak or not sk:
            raise RuntimeError(
                "Bedrock credentials missing. Go to Settings → select Bedrock → enter your AWS Access Key ID and Secret Access Key."
            )
        _client = AnthropicBedrock(
            aws_access_key=ak,
            aws_secret_key=sk,
            aws_region=rg,
        )
    else:
        _client = Anthropic(api_key=api_key)

def _classify_api_error(e):
    """Return a clean, actionable error message for common API failures."""
    msg = str(e).lower()
    if "insufficient_quota" in msg or "exceeded" in msg and "quota" in msg:
        return "Your API key has no credits remaining. Add billing/credits at your provider's dashboard and retry."
    if "invalid_api_key" in msg or "invalid api key" in msg or "incorrect api key" in msg:
        return "Invalid API key. Check it in Settings and try again."
    if "rate_limit" in msg or "rate limit" in msg or "too many requests" in msg:
        return "Rate limited by the API. Waiting and retrying..."
    if "no such host" in msg or "name or service not known" in msg or "connection refused" in msg:
        return "Cannot reach the API. Check your internet connection, VPN, or firewall."
    if "connection error" in msg or "connect timeout" in msg or "timed out" in msg:
        return "Cannot reach the API. Check your internet connection, VPN, or firewall."
    if "unrecognizedclientexception" in msg or "invalidclienttokenid" in msg:
        return "AWS credentials rejected. Check your Access Key ID and Secret in Railway variables."
    if "accessdeniedexception" in msg or "is not authorized to perform" in msg:
        return "AWS account doesn't have Bedrock access. Enable Bedrock model access in the AWS console."
    if "validationexception" in msg and "model" in msg:
        return "Bedrock model not found. Set BEDROCK_MODEL in Railway variables to a model you have access to."
    if "authentication" in msg or "401" in msg:
        return "Authentication failed. Your API key may be expired or revoked."
    return None

def ask(system, user, max_tokens=3000):
    last_err = None
    for attempt in range(1, 4):
        try:
            if _provider == "openai":
                r = _client.chat.completions.create(
                    model=_active_openai_model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                _token_usage["calls"] += 1
                if r.usage:
                    _token_usage["input"] += r.usage.prompt_tokens or 0
                    _token_usage["output"] += r.usage.completion_tokens or 0
                return r.choices[0].message.content.strip()
            else:
                model_id = BEDROCK_MODEL if _provider == "bedrock" else _active_claude_model
                # Bedrock on-demand requires cross-region inference profile prefix (us./eu./ap.)
                if _provider == "bedrock" and not model_id.startswith(("us.", "eu.", "ap.")):
                    model_id = f"us.{model_id}"
                r = _client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                _token_usage["calls"] += 1
                if hasattr(r, "usage"):
                    _token_usage["input"] += getattr(r.usage, "input_tokens", 0)
                    _token_usage["output"] += getattr(r.usage, "output_tokens", 0)
                return r.content[0].text.strip()
        except Exception as e:
            last_err = e
            clean = _classify_api_error(e)
            if clean and ("quota" in clean.lower() or "invalid" in clean.lower() or "authentication" in clean.lower()):
                raise RuntimeError(clean) from e
            if attempt < 3:
                time.sleep(1.5 * attempt)
    clean = _classify_api_error(last_err)
    raise RuntimeError(clean or f"LLM request failed after 3 attempts ({_provider}): {last_err}")

def extract_code(txt):
    if not txt:
        return ""
    try:
        if "```python" in txt:
            return txt.split("```python", 1)[1].split("```", 1)[0].strip()
        if "```" in txt:
            return txt.split("```", 1)[1].split("```", 1)[0].strip()
    except Exception:
        pass
    return txt.strip()

def classify_failure_reason(error_text: str) -> str:
    e = (error_text or "").lower()
    if "filenotfounderror" in e or "no such file or directory" in e:
        return "data_path"
    if "train.csv" in e or "test.csv" in e:
        return "data_path"
    if "validate_parameter_constraints" in e or "invalid parameter" in e:
        return "invalid_hyperparameter"
    if "absolute_deviation" in e:
        return "invalid_hyperparameter"
    if "no module named" in e:
        return "missing_package"
    if "timeout" in e:
        return "timeout"
    if "no json in stdout" in e:
        return "bad_output_format"
    if "json" in e and "decode" in e:
        return "bad_output_format"
    return "runtime_error"

def apply_code_guardrails(code: str) -> tuple[str, list[str]]:
    """Patch common LLM-generated breakages before execution."""
    fixed = code or ""
    notes: list[str] = []

    if "absolute_deviation" in fixed:
        fixed = fixed.replace("absolute_deviation", "absolute_error")
        notes.append("normalized_invalid_loss_name")

    # Fix: sklearn ≥1.4 removed the `squared` kwarg from mean_squared_error.
    # Replace mean_squared_error(..., squared=False) → np.sqrt(mean_squared_error(...))
    # Replace mean_squared_error(..., squared=True)  → mean_squared_error(...)
    def _fix_mse_squared(m):
        args = m.group(1)          # everything inside mse(...)
        # Remove squared=False or squared=True from the arg list
        cleaned = re.sub(r',?\s*squared\s*=\s*(True|False)\s*', '', args).strip().rstrip(',').strip()
        is_rmse = 'False' in m.group(0)  # squared=False means RMSE
        inner = f"mean_squared_error({cleaned})"
        if is_rmse:
            # Ensure numpy is available
            return f"np.sqrt({inner})"
        return inner

    if 'squared=' in fixed:
        new_fixed = re.sub(r'mean_squared_error\(([^)]+squared\s*=\s*(?:True|False)[^)]*)\)', _fix_mse_squared, fixed)
        if new_fixed != fixed:
            fixed = new_fixed
            # Make sure numpy is imported
            if 'import numpy' not in fixed and 'import numpy as np' not in fixed:
                fixed = 'import numpy as np\n' + fixed
            notes.append("fixed_mse_squared_kwarg")

    # Strip any DATA_PATH or TIME_BUDGET redefinitions — these are injected by execute()
    for var in ("DATA_PATH", "TIME_BUDGET"):
        pattern = rf'^{var}\s*=\s*.+$'
        new_fixed = re.sub(pattern, f"# {var} injected by 19Labs (removed duplicate)", fixed, flags=re.MULTILINE)
        if new_fixed != fixed:
            fixed = new_fixed
            notes.append(f"stripped_{var.lower()}_override")

    # Catch os.environ.get('DATA_PATH', ...) pattern — replace with bare DATA_PATH usage
    env_pattern = r"os\.environ\.get\(\s*['\"]DATA_PATH['\"].*?\)"
    if re.search(env_pattern, fixed):
        fixed = re.sub(env_pattern, "DATA_PATH", fixed)
        notes.append("replaced_environ_get_data_path")

    # Avoid hardcoded local paths; training script must always use injected DATA_PATH.
    patterns = [
        r"pd\.read_csv\(\s*r?[\"'](?:\./)?train\.csv[\"']\s*\)",
        r"pd\.read_csv\(\s*r?[\"'](?:\./)?test\.csv[\"']\s*\)",
        r"pd\.read_csv\(\s*r?[\"']data/train\.csv[\"']\s*\)",
        r"pd\.read_csv\(\s*r?[\"'](?:\./)?data\.csv[\"']\s*\)",
    ]
    for p in patterns:
        new_fixed = re.sub(p, "pd.read_csv(DATA_PATH, sep=DATA_SEP)", fixed)
        if new_fixed != fixed:
            fixed = new_fixed
            notes.append("replaced_hardcoded_csv_path")

    if "json.dumps(" in fixed and "import json" not in fixed and "from json import" not in fixed:
        fixed = "import json\n" + fixed
        notes.append("added_missing_json_import")

    return fixed, notes

def normalize_reliability_mode(mode: str | None) -> str:
    m = (mode or "balanced").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "safe": "demo_safe",
        "demo": "demo_safe",
        "demo_safe": "demo_safe",
        "balanced": "balanced",
        "normal": "balanced",
        "aggressive": "aggressive",
        "sota": "aggressive",
    }
    resolved = aliases.get(m, "balanced")
    return resolved if resolved in RELIABILITY_PROFILES else "balanced"

def _smart_default_metric(task, hint=""):
    """Pick the right metric based on the task type and user hint — never blindly default to RMSE."""
    hint_lo = (hint or "").lower()
    # If user explicitly mentions a metric, use it
    for m in ["mape", "smape", "mae", "rmse", "mse", "nse", "r2", "auc", "f1", "accuracy", "precision", "recall", "logloss"]:
        if m in hint_lo:
            return m
    task_lo = (task or "").lower()
    if "timeseries" in task_lo or "forecast" in task_lo:
        return "mape"  # MAPE is the business-standard for forecasting
    if "classif" in task_lo:
        if "binary" in task_lo:
            return "auc"
        return "f1"
    if "cluster" in task_lo:
        return "silhouette"
    if "sentiment" in task_lo or "nlp" in task_lo:
        return "f1"
    # Generic regression — R² is more interpretable than RMSE for business users
    return "rmse"

# ── INFER OBJECTIVE ────────────────────────────────────────────
def infer_objective(profile, hint="", domain_analysis="", previous_objective=None):
    # Build context-aware hint block so the LLM can interpret corrections correctly
    if hint and previous_objective:
        _prev_summary = (
            f"task={previous_objective.get('task','?')}, "
            f"target={previous_objective.get('target','?')}, "
            f"metric={previous_objective.get('metric','?')}, "
            f"direction={previous_objective.get('direction','?')}"
        )
        _hint_block = (
            f"\n⚠️  USER CORRECTION — you MUST interpret this in context of what was just proposed:\n"
            f"CURRENT PROPOSAL: {_prev_summary}\n"
            f"USER SAID: \"{hint}\"\n\n"
            f"This is a conversational reaction. Parse intent from context — do NOT take it literally:\n"
            f"  • 'no revenue forecast' when proposing predict_weight  → user wants revenue forecast instead\n"
            f"  • 'no, use NSE' when metric=rmse                       → change metric to NSE\n"
            f"  • 'no use MAE' when metric=rmse                        → change metric to MAE\n"
            f"  • 'no, churn' when target=revenue                      → change target to churn column\n"
            f"  • 'classification not regression' when task=Regression → change task to Classification\n"
            f"  • 'weekly not daily'                                    → adjust time granularity\n"
            f"  • 'add X as feature'                                    → note X as important feature\n"
            f"Apply the correction to the right field (task / target / metric / direction) and "
            f"keep everything else from the current proposal unchanged.\n"
        )
    elif hint:
        _hint_block = (
            f"\n⚠️  USER INSTRUCTION (overrides profile inferences):\n"
            f"The user said: \"{hint}\"\n"
            f"Use this to determine the correct target, task, and metric.\n"
        )
    else:
        _hint_block = ""

    # --- Hard-override: parse hint for an explicit target column name ---
    _forced_target = None
    if hint:
        _headers_lo = {h.lower(): h for h in profile["headers"]}
        # Pattern 1: TARGET="Churn" or TARGET='Churn' (from Kaggle context injection)
        _m = re.search(r'TARGET\s*=\s*["\']?(\w+)["\']?', hint, re.IGNORECASE)
        if _m and _m.group(1).lower() in _headers_lo:
            _forced_target = _headers_lo[_m.group(1).lower()]
        if not _forced_target:
            # Pattern 2: scan ALL words in the hint for any column name match.
            # If multiple columns are mentioned, prefer the one that is a known
            # target-like name or appears after a predict/target keyword.
            _TARGET_NAMES = {
                "churn","target","label","class","y","outcome","default","fraud",
                "survived","response","converted","clicked","purchased","defaulted",
                "attrition","cancelled","retained","approved","spam","toxic",
                "sentiment","result","status","success",
            }
            _hint_words = re.findall(r'\w+', hint.lower())
            # First pass: exact column name mentioned anywhere in hint
            _mentioned = [_headers_lo[w] for w in _hint_words if w in _headers_lo]
            if _mentioned:
                # Prefer known target names, then columns after "predict/target" keywords
                _known = [c for c in _mentioned if c.lower() in _TARGET_NAMES]
                if _known:
                    _forced_target = _known[0]
                else:
                    # Find column name that appears after "predict" / "predicting" / "forecast"
                    _predict_idx = next(
                        (i for i, w in enumerate(_hint_words)
                         if w in ("predict","predicting","forecast","forecasting","classify","classifying")),
                        None)
                    if _predict_idx is not None:
                        _after = [_headers_lo[w] for w in _hint_words[_predict_idx+1:]
                                  if w in _headers_lo]
                        _forced_target = _after[0] if _after else _mentioned[0]
                    else:
                        _forced_target = _mentioned[0]

    if _forced_target:
        # Also derive the task from the target column's type — don't let the LLM guess
        _target_col_info = next((c for c in profile.get('columns', []) if c['name'] == _forced_target), None)
        if _target_col_info and _target_col_info['type'] in ('categorical', 'high_cardinality'):
            _forced_task = "BinaryClassification" if _target_col_info['unique'] <= 2 else "MultiClassClassification"
        else:
            _forced_task = None
        _task_note = f"TASK must be {_forced_task} (target is categorical with {_target_col_info['unique'] if _target_col_info else '?'} classes).\n" if _forced_task else ""
        _override_block = (
            f"\n\U0001f6a8 MANDATORY OVERRIDE — DO NOT IGNORE:\n"
            f"The target column is CONFIRMED as \"{_forced_target}\".\n"
            f"You MUST set TARGET: {_forced_target}\n"
            f"{_task_note}"
            f"Setting TARGET to anything else (including 'id') is WRONG.\n"
        )
    else:
        _forced_task = None
        _override_block = ""

    resp = ask(
        "You are 19Labs. Given a deep expert domain analysis, produce precise ML objective parameters. "
        "When the user gives an instruction, interpret it in context of the current plan to find their true intent.",
        f"""{_override_block}Based on the expert domain analysis below, extract the exact ML objective parameters.
{_hint_block}
EXPERT DOMAIN ANALYSIS:
{domain_analysis or "(not available — reason from profile)"}

DATASET SUMMARY:
- Rows: {profile['rows']:,} | Cols: {profile['cols']}
- Columns: {', '.join(profile['headers'])}
- Signals: {'; '.join(profile.get('signals', []))}
- Target candidates: {', '.join(profile['target_candidates'])}

Reply ONLY in this exact format (no extra text):
DOMAIN: <specific domain>
TASK: <exact task — e.g. BinaryClassification, TimeSeriesForecasting, SentimentAnalysis, Clustering>
TARGET: <exact column name, or "unsupervised">
METRIC: <primary metric — e.g. auc, f1, rmse, mape, accuracy, rouge, silhouette>
DIRECTION: <lower_is_better|higher_is_better>
CONFIDENCE: <0.0-1.0>
REASONING: <1-2 sentences from the domain analysis>
GOOD_ENOUGH: <concrete threshold e.g. "AUC > 0.85", "RMSE < 5000">
HYPOTHESES:
1. <Specific model> — <specific reason from domain analysis>
2. <Specific model> — <specific reason from domain analysis>
3. <Specific model> — <specific reason from domain analysis>""", 700)

    def g(k):
        m = re.search(rf"^{k}\s*:\s*(.+)", resp, re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    tc = profile["target_candidates"]
    task_inferred = g("TASK") or "Regression"
    metric_from_llm = (g("METRIC") or "").strip().lower()

    # SMART METRIC OVERRIDE — don't let the LLM blindly pick RMSE for time series.
    # If user explicitly named a metric in their hint, that wins always.
    hint_lo = (hint or "").lower()
    user_metric = None
    for _m in ["mape","smape","wape","mae","rmse","nse","r2","auc","f1","accuracy","logloss","precision","recall"]:
        if re.search(rf'\b{_m}\b', hint_lo):
            user_metric = _m
            break

    if user_metric:
        metric = user_metric  # User explicitly asked → always honour
    elif "timeseries" in task_inferred.lower() or "forecast" in task_inferred.lower():
        # For time series, MAPE is the business-standard metric.
        # Only keep RMSE if LLM gave a strong reason AND user didn't mention MAPE.
        metric = "mape"
    elif metric_from_llm:
        metric = metric_from_llm
    else:
        metric = _smart_default_metric(task_inferred, hint)

    tc = profile["target_candidates"]
    return dict(domain=g("DOMAIN") or "General",
        task   =_forced_task or task_inferred,
        target =_forced_target or g("TARGET") or (tc[0] if tc else profile["headers"][-1]),
        metric =metric,
        direction=g("DIRECTION") or "lower_is_better",
        confidence=_safe_float(g("CONFIDENCE"), 0.7),
        reasoning =g("REASONING") or "",
        good_enough=g("GOOD_ENOUGH") or "",
        raw=resp)

def _extract_json_blob(txt):
    txt = txt.strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def discover_user_need(csv_path, user_hint="", previous_objective=None, api_key=None, provider="claude", model=None, companion_profiles=None):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("No API key.")
    _init_client(key, provider, model=model)

    # Support media (image/audio) directories in addition to CSV files
    _path = pathlib.Path(csv_path)
    if _path.is_dir():
        profile = profile_media_dataset(csv_path)
    else:
        profile = profile_dataset(csv_path)
    obj = infer_objective(profile, user_hint, previous_objective=previous_objective)

    # Build correction block — shows the agent what it WAS proposing so it can
    # correctly interpret ambiguous user corrections like "no revenue forecast"
    # (which means "no [to what you're doing], do revenue forecast")
    _prev_block = ""
    if previous_objective:
        _prev_block = (
            f"\nWHAT THE AGENT WAS PREVIOUSLY PROPOSING:\n"
            f"- task: {previous_objective.get('task','?')}\n"
            f"- target: {previous_objective.get('target','?')}\n"
            f"- metric: {previous_objective.get('metric','?')} ({previous_objective.get('direction','?')})\n"
            f"- reasoning: {previous_objective.get('reasoning','?')}\n"
        )

    _correction_block = ""
    if user_hint and previous_objective:
        _correction_block = (
            f"\n⚠️  USER CORRECTION — interpret as a reaction to the proposal above, not in isolation:\n"
            f"USER SAID: \"{user_hint}\"\n\n"
            f"Conversational interpretation guide:\n"
            f"  • 'no revenue forecast' when proposing predict_weight  → user wants revenue forecast\n"
            f"  • 'no use NSE' / 'use NSE instead' when metric=rmse    → change metric to NSE\n"
            f"  • 'no use MAE' when metric=rmse                        → change metric to MAE\n"
            f"  • 'no, churn' when target=revenue                      → change target to churn\n"
            f"  • 'classification not regression'                      → change task type\n"
            f"  • 'too complex' / 'simpler'                            → simplify approach\n"
            f"Apply ONLY what the user changed. Keep everything else from the current proposal.\n"
            f"Your recommended_objective and recommended_metric MUST reflect the correction.\n"
        )
    elif user_hint:
        _correction_block = (
            f"\n⚠️  USER INSTRUCTION:\n"
            f"The user said: \"{user_hint}\"\n"
            f"Your recommended_objective, recommended_metric, and plan MUST align with this.\n"
        )

    # Build structured FILES section for LLM prompt (shows all Kaggle files with roles)
    if companion_profiles:
        _train_name = pathlib.Path(csv_path).name
        _fl = [f"- {_train_name}: {profile['rows']:,} rows, {profile['cols']} cols,"
               f" headers: {profile['headers'][:10]} → TRAINING FILE (has target labels)"]
        for _fname, _fp in companion_profiles.items():
            if _fp["role"] == "test":
                _fl.append(f"- {_fname}: {_fp['rows']:,} rows, {_fp['cols']} cols,"
                           f" headers: {_fp['headers']} → TEST FILE (no target column, needs predictions)")
            else:
                _fl.append(f"- {_fname}: {_fp['rows']:,} rows, {_fp['cols']} cols,"
                           f" headers: {_fp['headers']} → SUBMISSION FORMAT (required output columns)")
        _files_block = "FILES (all datasets in this competition):\n" + "\n".join(_fl)
    else:
        _files_block = ""

    advice_raw = ask(
        "You are a sharp, experienced ML engineer having a casual conversation with a colleague. "
        "You think fast, notice what matters, and skip the filler. You never say 'I scanned your dataset' "
        "or start with a generic status update. You lead with insight. "
        "Every dataset has something specific and interesting — always find the unique angle for THIS data.",
        f"""Analyze this dataset setup and respond naturally.
{_prev_block}{_correction_block}
Return STRICT JSON with keys:
{{
  "recommended_objective": "short objective sentence",
  "recommended_metric": "metric name",
  "clarifying_questions": ["q1", "q2", "q3"],
  "decision_tree": [
    {{
      "id": "short_snake_case_id",
      "question": "single focused question",
      "options": ["option 1", "option 2", "option 3"]
    }}
  ],
  "experiment_directions": ["direction1", "direction2", "direction3"],
  "risks": ["risk1", "risk2"],
  "first_iteration_plan": "one concise paragraph",
  "agent_message": "2-3 sentences MAX. Rules: (1) Pick ONE specific, concrete observation about THIS dataset — a column name, a class imbalance ratio, an unusual feature count, a datetime signal, whatever is most notable — and lead with it naturally. (2) State what you'll do about it. (3) End with one short invite. FORBIDDEN openers: 'I scanned', 'I analyzed', 'I found', 'Looking at', 'This dataset', 'The dataset'. FORBIDDEN phrases: 'regression baseline' for classification tasks. For Kaggle competitions: open by naming what train/test/submission are for and the specific prediction target. Make every response feel fresh and specific to THIS data — never a template."
}}

DATA PROFILE (train file):
- rows: {profile['rows']}
- cols: {profile['cols']}
- headers: {profile['headers']}
- numeric: {profile['numeric']}
- categorical: {profile['categorical']}
- datetime: {profile['datetime']}
- text_columns: {profile.get('text', [])}
- signals: {profile.get('signals', [])}
{_files_block}
COMPETITION CONTEXT: {"KAGGLE COMPETITION — train on train.csv, generate predictions for test.csv, save submission.csv" if ("competition" in (user_hint or "").lower() or "kaggle" in (user_hint or "").lower()) else "standard dataset"}

INFERRED OBJECTIVE:
- task: {obj['task']}
- target: {obj['target']}
- metric: {obj['metric']} ({obj['direction']})
- domain: {obj['domain']}
- reasoning: {obj['reasoning']}
""",
        1500
    )
    advice = _extract_json_blob(advice_raw)

    # ── Minimal post-process: only fix factually wrong task language + strip generic openers ──
    if isinstance(advice, dict):
        _msg = advice.get("agent_message", "") or ""
        _task = obj.get("task", "")
        # Strip generic dataset-scan openers if the LLM generates them anyway
        _msg = re.sub(
            r"^(I['']ve scanned your dataset[^.]*\.\s*|I analyzed your data[^.]*\.\s*|"
            r"I found [0-9,]+ rows[^.]*\.\s*|After (scanning|analyzing)[^.]*\.\s*)",
            "", _msg, flags=re.IGNORECASE).lstrip()
        if "classif" in _task.lower():
            _msg = re.sub(r'\bregression baseline\b', 'classification baseline', _msg, flags=re.IGNORECASE)
            _msg = re.sub(r'\bregression model\b', 'classification model', _msg, flags=re.IGNORECASE)
            _msg = re.sub(r'\bstart with a regression\b', 'start with a classification', _msg, flags=re.IGNORECASE)
        advice["agent_message"] = _msg
    return {
        "profile": profile,
        "objective": obj,
        "discovery": advice,
        "raw": advice_raw,
    }


# ── MODEL NAME INFERENCE ──────────────────────────────────────
_MODEL_PATTERNS = [
    (r'(?:XGBRegressor|XGBClassifier|xgb\.XGB)', 'XGBoost'),
    (r'(?:LGBMRegressor|LGBMClassifier|lgb\.LGBMModel)', 'LightGBM'),
    (r'(?:CatBoostRegressor|CatBoostClassifier)', 'CatBoost'),
    (r'(?:RandomForestRegressor|RandomForestClassifier)', 'Random Forest'),
    (r'(?:GradientBoostingRegressor|GradientBoostingClassifier)', 'Gradient Boosting'),
    (r'(?:ExtraTreesRegressor|ExtraTreesClassifier)', 'Extra Trees'),
    (r'(?:AdaBoostRegressor|AdaBoostClassifier)', 'AdaBoost'),
    (r'(?:BaggingRegressor|BaggingClassifier)', 'Bagging'),
    (r'(?:StackingRegressor|StackingClassifier)', 'Stacking Ensemble'),
    (r'(?:VotingRegressor|VotingClassifier)', 'Voting Ensemble'),
    (r'Ridge\(', 'Ridge'),
    (r'Lasso\(', 'Lasso'),
    (r'ElasticNet\(', 'ElasticNet'),
    (r'LinearRegression\(', 'Linear Regression'),
    (r'LogisticRegression\(', 'Logistic Regression'),
    (r'SVR\(|SVC\(', 'SVM'),
    (r'KNeighborsRegressor|KNeighborsClassifier', 'KNN'),
    (r'DecisionTreeRegressor|DecisionTreeClassifier', 'Decision Tree'),
    (r'MLPRegressor|MLPClassifier', 'Neural Net (MLP)'),
    (r'GaussianNB|BernoulliNB|MultinomialNB', 'Naive Bayes'),
    (r'HuberRegressor', 'Huber Regressor'),
    (r'SGDRegressor|SGDClassifier', 'SGD'),
    (r'BayesianRidge', 'Bayesian Ridge'),
    (r'(?:keras|tensorflow|tf\.)', 'TensorFlow/Keras'),
    (r'(?:torch|nn\.Module)', 'PyTorch'),
]

def _infer_model_name(code: str) -> str:
    """Extract most likely model name from train.py source code."""
    found = []
    for pattern, name in _MODEL_PATTERNS:
        if re.search(pattern, code):
            found.append(name)
    return found[0] if found else "Custom Model"


# ── EXECUTE (Karpathy-style: redirect to run.log, grep metrics) ─
def execute(code, csv_path, ws, exp_num, data_sep=","):
    ws = pathlib.Path(ws)
    script = ws / "train.py"
    run_log = ws / "run.log"
    full = f"DATA_PATH = {repr(str(csv_path))}\nDATA_SEP = {repr(data_sep)}\nTIME_BUDGET = {TIME_BUDGET}\n\n{code}"
    script.write_text(full)

    # Also save numbered copy for audit trail
    (ws / f"exp_{exp_num:02d}.py").write_text(full)

    t0 = time.time()
    try:
        # Redirect ALL output to run.log — never flood context (Karpathy rule)
        with open(run_log, "w") as log_f:
            p = subprocess.run(
                [sys.executable, str(script)],
                stdout=log_f, stderr=subprocess.STDOUT,
                timeout=EXEC_TIMEOUT, cwd=str(ws),
            )
        elapsed = time.time() - t0
        stdout = run_log.read_text() if run_log.exists() else ""

        if p.returncode != 0:
            # Read tail of log for error context (like `tail -n 50 run.log`)
            tail = "\n".join(stdout.split("\n")[-50:])
            return dict(success=False, error=tail[-600:], elapsed=elapsed, stdout=stdout)

        # Grep metrics from log (like `grep "^val_bpb:" run.log`)
        for line in reversed(stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return dict(success=True, metrics=json.loads(line), stdout=stdout, elapsed=elapsed)
                except Exception:
                    pass

        has_metrics = bool(re.search(r'(?:rmse|r2|mape|mae|nse|auc|f1|accuracy|mse)\s*[:=]\s*[\d.]+', stdout, re.IGNORECASE))
        return dict(success=False,
            error=f"No JSON in stdout. Tail: {stdout[-300:]}",
            elapsed=elapsed, stdout=stdout, fixable_output=has_metrics)
    except subprocess.TimeoutExpired:
        # Kill on timeout — treat as failure (Karpathy: if >10min, kill and discard)
        return dict(success=False, error=f"Timeout {EXEC_TIMEOUT}s — killed", elapsed=EXEC_TIMEOUT)
    except Exception as e:
        return dict(success=False, error=str(e), elapsed=time.time() - t0)

# ── AUTO-FIX ───────────────────────────────────────────────────
def auto_fix(code, error):
    fixed = extract_code(ask(
        "You are 19Labs debugger. Fix failing Python ML scripts. Output ONLY a ```python block.",
        f"""Fix this script. Any package you import will be auto-installed — do not downgrade the approach.

ERROR:
{error[:800]}

SCRIPT:
```python
{code[:2500]}
```

RULES:
- Load data only with DATA_PATH: `df = pd.read_csv(DATA_PATH, sep=DATA_SEP)` (DATA_SEP is pre-injected; or use appropriate loader for the data type)
- NEVER hardcode 'train.csv' or any local absolute path.
- ALWAYS split data into train/test. Compute metrics on BOTH.
- Final line: print(json.dumps(metrics)) — MUST include "model", plus train_ and test_ prefixed metrics.
- If the error is a missing package, keep the import — it will be auto-installed before the next run.

Fix the error. Keep the model/approach if possible. Output ONLY fixed ```python code.""", 6000))
    fixed, _ = apply_code_guardrails(fixed)
    return fixed


def write_program_md(profile, obj, history, insights, domain_analysis=""):
    hist_txt = "\n".join(
        f"- Exp {h.get('num', 0):02d}: status={h.get('status', 'unknown')} "
        f"{h.get('metric_name', obj.get('metric', 'metric'))}={h.get('metric_val', 0):.6f} "
        f"model={h.get('model', 'unknown')} note={h.get('note', '')[:140]}"
        for h in history
    ) if history else "- (no experiments yet)"
    ins_txt = "\n".join(f"- {x}" for x in insights) if insights else "- (none yet)"

    program = ask(
        "You are an autonomous ML research lead. Write high-signal research specs in markdown.",
        f"""Write a complete `program.md` for this project. Rewrite fully from scratch each time.

ENVIRONMENT:
- Auto-install is available. Any pip-installable package can be imported.
- Pre-installed: sklearn, xgboost, lightgbm, catboost, pandas, numpy, matplotlib, scipy, statsmodels, optuna, shap, joblib
- Any missing package is auto-installed on demand via pip. Use whatever you need.
- Use whatever is genuinely best for the task. Do NOT artificially limit yourself.

KNOWN API BREAKAGES — avoid these exactly:
- sklearn ≥1.4: mean_squared_error() has NO `squared` kwarg. Use np.sqrt(mean_squared_error(y,p)) for RMSE.
- sklearn ≥1.2: use class_weight='balanced' not balanced_subsample for non-forests.
- pandas ≥2.0: DataFrame.append() removed — use pd.concat([df, new_row.to_frame().T]).

EXPERT DOMAIN ANALYSIS (written by a senior data scientist — treat as ground truth):
{domain_analysis or "(not available)"}

OBJECTIVE:
- Task: {obj.get('task', 'Regression')}
- Target: {obj.get('target', '')}
- Metric: {obj.get('metric', 'rmse')} ({obj.get('direction', 'lower_is_better')})
- Domain: {obj.get('domain', 'General')}
- Good enough: {obj.get('good_enough', '')}
- User hint: {obj.get('user_hint', '')}
- Reliability mode: {obj.get('reliability_mode', 'balanced')}
- Execution policy: {obj.get('execution_policy', 'Balance reliability and performance.')}

EXPERIMENT HISTORY:
{hist_txt}

LEARNINGS:
{ins_txt}

KARPATHY DISCIPLINE (MANDATORY):
- train.py is the ONLY file you may edit. ALL code goes in train.py.
- prepare.py is READ-ONLY. Never reference or modify it.
- DATA_PATH and TIME_BUDGET are pre-defined variables injected at line 1. DO NOT redefine them. DO NOT use os.environ.
- TIME_BUDGET = {obj.get('time_budget', TIME_BUDGET)}s. Add wall-clock checks in training loops.
- ALWAYS split data into train/test. Compute metrics on BOTH and report both.
- ALL output goes to stdout. Metrics MUST be printed as a single JSON line at the end with ALL applicable metrics:
  print(json.dumps({{"model": "ModelName", "train_rmse": val, "test_rmse": val, "train_r2": val, "test_r2": val, "rmse": test_val, "r2": test_val, "mape": val, "mae": val, "nse": val}}))
  The "model" key is MANDATORY. ALWAYS include train_ and test_ prefixed metrics for at least rmse and r2.
- NEVER print sentinel values (999, -1, 0) as fallback metrics. If training fails, let the exception propagate — the system handles crashes.
- KEEP means the experiment improved the metric. DISCARD means it didn't. This is enforced by git.
- Each experiment = 1 git commit. DISCARD = `git reset --hard` to the last KEEP. You CANNOT recover discarded code.
- NEVER STOP iterating. If stagnating, RADICALLY change strategy (different algorithm, different features, different preprocessing).
- If your code crashes, analyze the error, fix the CAUSE (not the symptom), and move on.

Your `program.md` MUST include these sections:
1) Mission
2) Task + metric definition (including exact KEEP criteria)
3) Constraints (packages, robustness, determinism, time budget)
4) What worked
5) What failed and why
6) Current hypothesis for next iteration
7) Plan for next train.py rewrite

Output ONLY markdown for `program.md`.""",
        2500
    )
    return program.strip()

def write_train_py(program_md, profile, obj, exp_num, history, domain_analysis=""):
    hist_txt = "\n".join(
        f"- Exp {h.get('num', 0):02d}: {h.get('status', 'unknown')} "
        f"{h.get('metric_name', obj.get('metric', 'metric'))}={h.get('metric_val', 0):.6f}"
        for h in history
    ) if history else "- (none)"

    # Pre-compute Kaggle competition block
    _is_kaggle = bool(obj.get('is_kaggle'))
    _kaggle_test = obj.get('kaggle_test_file') or ''
    _kaggle_sample = obj.get('kaggle_sample_file') or ''
    if _is_kaggle and _kaggle_test:
        _sample_line = (
            f"  sample_sub = pd.read_csv(os.path.join(os.path.dirname(DATA_PATH), {repr(_kaggle_sample)}))"
            if _kaggle_sample else
            "  # Read sample_submission.csv to get the required output column names"
        )
        _kaggle_train_block = f"""
KAGGLE COMPETITION MODE — MANDATORY RULES:
1. DO NOT do a simple train_test_split for evaluation. Instead use StratifiedKFold (classification)
   or KFold (regression) cross-validation on the full train.csv to estimate performance.
2. After CV, retrain the FINAL model on ALL of train.csv (no holdout withheld).
3. Load {repr(_kaggle_test)} separately — this is the unlabelled test set, NEVER train on it.
4. Apply the EXACT same preprocessing pipeline (fitted on train only) to the test set.
5. Generate predictions for test set and save submission.csv:
  import os
  test_path = os.path.join(os.path.dirname(DATA_PATH), {repr(_kaggle_test)})
  test_df = pd.read_csv(test_path)
  test_features = preprocessor.transform(test_df[feature_cols])
  test_preds = final_model.predict(test_features)
{_sample_line}
  submission = pd.DataFrame({{sample_sub.columns[0]: test_df[sample_sub.columns[0]], sample_sub.columns[-1]: test_preds}})
  submission.to_csv('submission.csv', index=False)
  print(f"submission.csv saved — {{len(submission)}} rows, columns: {{list(submission.columns)}}")
"""
    else:
        _kaggle_train_block = ""

    # Pre-compute media-specific strings (avoids backslash-in-f-string issues on Python < 3.12)
    _is_media = bool(profile.get('is_media'))
    if _is_media:
        _media_line = (
            f"- Media type: {profile['media_type']} | Files: {profile['total_files']}"
            f" | Classes: {list(profile['classes'].keys())}"
        )
        _data_load_line = (
            "- DATA_PATH is a DIRECTORY path containing class subfolders of "
            + profile.get('media_type', '') + " files"
            " (e.g. DATA_PATH/cats/*.jpg, DATA_PATH/dogs/*.jpg)."
            " Load with glob/PIL/librosa — NOT pd.read_csv."
        )
    else:
        _media_line = ""
        _data_load_line = (
            "- `df = pd.read_csv(DATA_PATH, sep=DATA_SEP)` — this is the ONLY way to load data."
            " DATA_SEP is already set to the correct delimiter (e.g. ',' or ';' or '\\t')."
        )

    code = ask(
        "You write complete production-grade Python training scripts. "
        "You are a senior ML engineer — you know exactly what approach fits each dataset type and domain. "
        "You never write generic code when expert-level code is possible.",
        f"""Write `train.py` — experiment {exp_num} — for this specific domain and task.

ENVIRONMENT:
- Auto-install available for any pip-installable package.
- Pre-installed: sklearn, xgboost, lightgbm, catboost, pandas, numpy, matplotlib, scipy, statsmodels, optuna, shap, joblib
- Any missing package is auto-installed on demand via pip. Use whatever you need.
- CRITICAL: Use what genuinely fits. For NLP → transformers (HuggingFace). For time series → prophet/statsmodels.
  For imbalanced → SMOTE+class weights. For tabular → gradient boosting + optuna. For images → torch/tensorflow.

KNOWN API BREAKAGES — these will crash, avoid exactly:
- mean_squared_error(y, p, squared=False) → WRONG. Use: np.sqrt(mean_squared_error(y, p))
- mean_squared_error(y, p, squared=True)  → WRONG. Use: mean_squared_error(y, p)
- DataFrame.append() → REMOVED in pandas 2.0. Use pd.concat([df, row.to_frame().T], ignore_index=True)

EXPERT DOMAIN ANALYSIS (source of truth — follow this strategy):
{domain_analysis[:2000] if domain_analysis else "(not available — use program spec)"}

EXPERIMENT: {exp_num}
TASK: {obj.get('task', 'Regression')} | TARGET: {obj.get('target', '')}
METRIC: {obj.get('metric', 'rmse')} ({obj.get('direction', 'lower_is_better')})

RECENT HISTORY:
{hist_txt}

PROGRAM SPEC (source of truth):
```markdown
{program_md[:12000]}
```

DATA PROFILE:
- Rows: {profile['rows']:,}
- Cols: {profile['cols']}
- Headers: {', '.join(profile['headers'])}
- Text columns: {', '.join(profile.get('text', [])) or 'none'}
- Signals: {'; '.join(profile.get('signals', []))}
{_media_line}

EXECUTION POLICY:
- Reliability mode: {obj.get('reliability_mode', 'balanced')}
- {obj.get('execution_policy', 'Balance reliability and performance.')}

KARPATHY DISCIPLINE (MANDATORY):
- DATA_PATH and TIME_BUDGET are Python variables pre-defined BEFORE your code runs (injected at line 1).
  DO NOT redefine them. DO NOT use os.environ.get(). Use them directly.
{_data_load_line}
- TIME_BUDGET is also pre-defined. DO NOT redefine it. Your entire training MUST complete within it.
  Add a wall-clock check: `import time; _start = time.time()` at top, and periodically
  check `if time.time() - _start > TIME_BUDGET * 0.9: break` in any training loops.
  This is NON-NEGOTIABLE. Timeout = automatic DISCARD.
- Robust preprocessing (nulls, categoricals, datetime).
- Deterministic behavior (set random seeds).
- Save model via `joblib.dump(model, 'model.pkl')`.
- PRIMARY METRIC: {obj.get('metric','rmse').upper()} — optimize for THIS, not RMSE.
  Use as eval_metric in LightGBM/XGBoost/CatBoost. Use as scoring in cross_val_score.
- MANDATORY predictions.csv — ALWAYS save this file after fitting, no exceptions:
  ```python
  import pandas as pd
  _pred_df_rows = []
  for i, (act, pred) in enumerate(zip(y_test, y_pred)):
      row = {{'actual': float(act), 'predicted': float(pred)}}
      # If you have a date column, add it:  row['date'] = str(test_dates.iloc[i]) if hasattr(test_dates,'iloc') else str(i)
      _pred_df_rows.append(row)
  pd.DataFrame(_pred_df_rows).to_csv('predictions.csv', index=False)
  ```
  This table is shown to the user in the Results tab — it MUST exist.
- ALL output goes to stdout. Final line MUST be `print(json.dumps(metrics))` where metrics is a dict:
  REQUIRED keys:
  - "model": string (e.g. "LightGBM")
  - "{obj.get('metric', 'mape')}": float — PRIMARY metric, TEST set
  ALWAYS INCLUDE:
  - "train_{obj.get('metric','mape')}": float, "test_{obj.get('metric','mape')}": float
  - "train_rmse": float, "test_rmse": float
  - "train_mape": float, "test_mape": float
  - "train_r2": float, "test_r2": float
  - "rmse": float, "mape": float, "mae": float, "r2": float, "nse": float
  - "what_worked": string
  Example for MAPE primary: print(json.dumps({{"model":"LightGBM","mape":0.12,"train_mape":0.08,"test_mape":0.12,"rmse":4500.0,"train_rmse":3200.0,"test_rmse":4500.0,"r2":0.91,"train_r2":0.97,"test_r2":0.91,"mae":2100.0,"nse":0.89,"what_worked":"lag features"}}))
  NEVER catch exceptions around the metrics print.
- GENERATE PLOTS: After training, save ALL of the following plots using matplotlib (Agg backend).
  Apply this dark style before any plot:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    BG='#09090b'; BLUE='#3b82f6'; PURPLE='#8b5cf6'; GREEN='#34d399'; RED='#ef4444'; DIM='#27272a'; MUTED='#3f3f46'
    plt.rcParams.update({{'figure.facecolor':BG,'axes.facecolor':BG,'axes.spines.top':False,
      'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':True,
      'axes.edgecolor':DIM,'text.color':'#fafafa','xtick.color':'#52525b','ytick.color':'#52525b',
      'axes.grid':True,'grid.color':'#18181b','grid.linewidth':0.8,'legend.facecolor':BG,'legend.edgecolor':DIM,'legend.fontsize':9}})
  def _save(fig, name):
    plt.tight_layout(pad=1.6); fig.savefig(name, dpi=160, facecolor=BG, bbox_inches='tight'); plt.close(fig)

- SAVE PREDICTIONS CSV (MANDATORY): After fitting, always save:
  pred_df = pd.DataFrame({{'actual': y_test.values if hasattr(y_test,'values') else list(y_test), 'predicted': list(y_pred)}})
  # If a date/time column was found, insert it as first column: pred_df.insert(0, 'date', test_dates_values)
  pred_df.to_csv('predictions.csv', index=False)
  This enables the forecast comparison table in the UI. Never skip this.
{_kaggle_train_block}
- OPTIMIZE FOR THE PRIMARY METRIC: {obj.get('metric','rmse').upper()} — not RMSE unless that IS the metric.
  In LightGBM: metric='{obj.get('metric','rmse')}' in params. In XGBoost: eval_metric='{obj.get('metric','rmse')}'.
  In sklearn GridSearchCV/cross_val_score: scoring='neg_{obj.get('metric','rmse')}' or a custom scorer.

  1. timeseries.png — THE MAIN PLOT. Full time series of actual vs predicted over the ENTIRE date range.
     - Detect date/time column (parse with pd.to_datetime). If no date column, use index as x-axis.
     - Sort by date. Plot ACTUALS as a solid white/light line (lw=1.5, alpha=0.9, color='#fafafa', label='actual').
     - Shade the TRAIN region with ax.axvspan (alpha=0.06, color=BLUE, label='train').
     - Plot TEST PREDICTIONS as a bold blue line (lw=2.5, color=BLUE, label='predicted', zorder=5).
     - If forecasting future steps, extend with a dashed line (linestyle='--', color=GREEN).
     - Add a thin vertical line at train/test boundary (color=DIM, lw=1, linestyle=':').
     - Title: f"{{target_col}}  ·  forecast vs actual" left-aligned. Hide y-axis label, keep x ticks.
     - Legend top-right. fig size (14, 4).

  2. correlation.png — Feature correlation heatmap.
     - Compute df[numeric_cols].corr(). Keep top 20 features by abs correlation with target.
     - Use: import seaborn as sns (or fall back to plt.imshow if no seaborn).
     - sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
         linewidths=0.4, linecolor='#18181b', annot_kws={{'size':7}}, ax=ax)
     - ax.set_facecolor(BG); colorbar text color '#71717a'. Title "feature correlations". fig size (10, 8).

  3. shap.png — SHAP feature importance (preferred) OR fallback to feature importance.
     - Try: import shap; explainer = shap.Explainer(model, X_train[:200]); sv = explainer(X_test[:200])
     - shap.summary_plot(sv, X_test[:200], plot_type='bar', show=False, color=BLUE)
     - plt.gcf().set_facecolor(BG); plt.gca().set_facecolor(BG); _save(plt.gcf(), 'shap.png')
     - FALLBACK if shap fails or model has no SHAP support: use model's feature_importances_ or coef_,
       plot top-20 horizontal bars (color=BLUE, height=0.45), invert y, hide x-axis, title="feature importance".

  4. predictions.png — Actual vs predicted scatter on TEST SET only.
     - scatter(y_test, y_pred, s=16, alpha=0.5, color=BLUE, linewidths=0).
     - Perfect line: ax.plot([mn,mx],[mn,mx], color=DIM, lw=1).
     - Title: "actual vs predicted  ·  test set". Hide y-axis. fig size (7, 6).

  5. residuals.png — Residual distribution (y_test - y_pred).
     - histogram bins=50, color=BLUE, alpha=0.85, edgecolor='none'.
     - vline at 0 (color=RED, lw=1.5). Title "residuals  ·  test set". Hide y-axis.

  Save in try/except so plot failures never block metrics output.
  If plot generation fails, pass silently — metrics are not optional, plots are.
- This is the ONLY file you edit. Everything must be self-contained in this script.

Output ONLY a complete ```python block.""",
        8000
    )
    clean, _ = apply_code_guardrails(extract_code(code))
    # Truncation guard: if code looks cut off, retry with simplified prompt
    if clean and not re.search(r'print\s*\(\s*json\.dumps', clean):
        import sys as _sys
        print("[engine] write_train_py: code appears truncated (no metrics print). Retrying with concise prompt.", file=_sys.stderr)
        code2 = ask(
            "You write concise, complete Python ML training scripts. Every script MUST end with print(json.dumps(metrics)).",
            f"""Write a SHORT complete train.py for experiment {exp_num}.

TASK: {obj.get('task')} | TARGET: {obj.get('target')} | METRIC: {obj.get('metric')}
DATA: DATA_PATH='{obj.get('data_path','data.csv')}' | {profile.get('rows')} rows | cols: {', '.join(profile.get('headers',[])[:15])}

Rules:
- Load data from DATA_PATH (top of script: DATA_PATH = r'{obj.get('data_path','data.csv')}')
- Train/test split. Compute metrics on test set.
- joblib.dump(model, 'model.pkl')
- LAST LINE: print(json.dumps({{"model":"ModelName","{obj.get('metric','rmse')}":float_val,"r2":float_val,"train_{obj.get('metric','rmse')}":float_val,"test_{obj.get('metric','rmse')}":float_val}}))
- Keep plots minimal — only predictions.png and residuals.png
- Keep total script under 120 lines

Output ONLY ```python block.""",
            6000
        )
        clean2, _ = apply_code_guardrails(extract_code(code2))
        if clean2 and re.search(r'print\s*\(\s*json\.dumps', clean2):
            clean = clean2
    return clean

def revise_after_iteration(program_md, train_py, score, error, history, domain_analysis="", obj=None):
    hist_txt = "\n".join(
        f"- Exp {h.get('num', 0):02d}: {h.get('status', 'unknown')} "
        f"{h.get('metric_name', 'metric')}={h.get('metric_val', 0):.6f} note={h.get('note', '')[:120]}"
        for h in history[-12:]
    ) if history else "- (none)"

    review = ask(
        "You are an autonomous ML researcher and senior data scientist. "
        "You make sharp KEEP/DISCARD decisions and rewrite experiments with genuine domain expertise.",
        f"""Given the current program.md, train.py, run result, and expert domain analysis — decide KEEP/DISCARD and write the next experiment.

EXPERT DOMAIN ANALYSIS (use this to guide your next approach):
{domain_analysis[:3000] if domain_analysis else "(not available)"}

KEEP CRITERIA:
- KEEP if the primary metric improved vs previous best in history.
- KEEP the first successful experiment always (baseline).
- DISCARD if metric is worse or equal to a previously KEPT experiment.

CRITICAL — NEXT TRAIN.PY RULES:
- MUST try a genuinely different approach every time. Never copy the previous script.
- Follow the domain analysis strategy: if it says use SMOTE for imbalance, do it. If it says walk-forward for time series, do it.
- Escalate sophistication: baseline → tuned → ensemble → domain-specific advanced approach.
- For NLP: progress from TF-IDF → sentence-transformers → fine-tuned transformer.
- For time series: progress from naive → statsmodels → prophet → LSTM.
- For tabular: progress from single model → optuna-tuned → stacking ensemble → neural tabular.
- NEVER downgrade the approach (no going from XGBoost back to linear regression).

Respond in EXACT format:
KEEP: <YES|NO>
REASONING: <short technical rationale>
PROGRAM_MD:
```markdown
<full rewritten program.md>
```
TRAIN_PY:
```python
<full rewritten train.py>
```

TRAIN.PY HARD RULES (Karpathy discipline):
- DATA_PATH, DATA_SEP, and TIME_BUDGET are pre-defined variables (injected at line 1 before your code).
  DO NOT redefine them. DO NOT use os.environ.get(). Use them directly: `df = pd.read_csv(DATA_PATH, sep=DATA_SEP)`
- Training MUST complete within TIME_BUDGET. Add wall-clock checks.
- ALWAYS split data into train/test. Compute metrics on BOTH sets.
- PRIMARY METRIC IS: {(obj or {}).get('metric', 'rmse').upper()} (direction: {(obj or {}).get('direction', 'lower_is_better')})
  YOU MUST OPTIMIZE FOR THIS METRIC — NOT RMSE unless that IS the primary metric.
  - Use it as eval_metric in LightGBM/XGBoost/CatBoost where supported
  - Use it as the scoring function in cross-validation / GridSearchCV
  - Use it to select the best model/hyperparameters
  - The primary metric value MUST be the key "{(obj or {}).get('metric', 'rmse')}" in your final JSON
- Final line: print(json.dumps(metrics)) — MUST include "model" key PLUS train_ and test_ prefixed metrics.
  Required keys: "{(obj or {}).get('metric', 'rmse')}" (primary, test set), "train_{(obj or {}).get('metric', 'rmse')}", "test_{(obj or {}).get('metric', 'rmse')}", "train_rmse", "test_rmse", "train_r2", "test_r2", "rmse", "r2", and any applicable: mape, mae, nse.
- MANDATORY predictions.csv — save this BEFORE the final print(), no exceptions:
  try:
    _rows = []
    for i in range(len(y_test)):
      _r = {{'actual': float(y_test.iloc[i] if hasattr(y_test,'iloc') else y_test[i]),
             'predicted': float(y_pred[i])}}
      # Add date if you have it: _r['date'] = str(test_dates.iloc[i])
      # Add customer/group if multi-series: _r['customer'] = str(test_group[i])
      _rows.append(_r)
    import pandas as _pd2; _pd2.DataFrame(_rows).to_csv('predictions.csv', index=False)
  except Exception: pass
  This file powers the Results tab showing actuals vs predicted to the user.
- GENERATE PLOTS (matplotlib, Agg backend). Same 5 plots as always:
    BG='#09090b'; BLUE='#3b82f6'; RED='#ef4444'; DIM='#27272a'
    plt.rcParams.update({{'figure.facecolor':BG,'axes.facecolor':BG,'axes.spines.top':False,
      'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':True,
      'axes.edgecolor':DIM,'text.color':'#fafafa','xtick.color':'#52525b','ytick.color':'#52525b',
      'axes.grid':True,'grid.color':'#18181b','grid.linewidth':0.8,'legend.facecolor':BG,'legend.edgecolor':DIM}})
  1. timeseries.png — full date-range actual vs predicted. Sort by date col. Shade train region (axvspan BLUE,alpha=0.06).
     Actuals: solid #fafafa line. Test predictions: bold BLUE line. Vertical split line: DIM dotted.
     Title "{{target_col}}  ·  forecast vs actual". fig (14,4).
  2. correlation.png — seaborn heatmap of top-20 numeric feature correlations, cmap='RdBu_r', annot=True fmt='.2f'.
  3. shap.png — shap.summary_plot bar if shap available, else feature_importances_ horizontal bars (top-20, BLUE, height=0.45, invert y).
  4. predictions.png — scatter actual vs predicted test set (s=16,alpha=0.5,BLUE), perfect line DIM.
  5. residuals.png — histogram residuals (bins=50,BLUE,alpha=0.85), vline 0 RED.
  Wrap ALL plot code in try/except — metrics must never fail because of plots.
- This is a git-tracked experiment. KEEP = commit. DISCARD = git reset --hard.
  You are writing the NEXT experiment. If you KEEP, your code becomes the new baseline.
  If you DISCARD, the codebase reverts to the last KEEP.
- NEVER repeat a failed strategy. ALWAYS change something significant when writing new train.py.
- If stagnating, RADICALLY change approach (different model, features, preprocessing).

LATEST SCORE JSON:
{json.dumps(score, indent=2, default=str) if score is not None else "null"}

LATEST ERROR:
{error or "none"}

RECENT HISTORY:
{hist_txt}

CURRENT PROGRAM.MD:
```markdown
{program_md[:12000]}
```

CURRENT TRAIN.PY:
```python
{train_py[:12000]}
```
""",
        10000
    )

    keep = bool(re.search(r"KEEP\s*:\s*YES", review, re.IGNORECASE))
    reason_match = re.search(r"REASONING\s*:\s*(.+)", review, re.IGNORECASE)
    reasoning = reason_match.group(1).strip() if reason_match else ""

    pm = ""
    tm = ""
    review_upper = review.upper()
    if "PROGRAM_MD:" in review_upper and "TRAIN_PY:" in review_upper:
        try:
            pm_part = review.split("PROGRAM_MD:", 1)[1].split("TRAIN_PY:", 1)[0]
            tm_part = review.split("TRAIN_PY:", 1)[1]
            pm = extract_code(pm_part)
            tm = extract_code(tm_part)
        except Exception:
            pass

    used_fallback = []
    if not pm:
        pm = program_md
        used_fallback.append("program_md")
    if not tm:
        tm = train_py
        used_fallback.append("train_py")
    if used_fallback:
        import sys as _sys
        print(f"[engine] revise_after_iteration: LLM fallback for {used_fallback} — response may be malformed", file=_sys.stderr)
    tm, _ = apply_code_guardrails(tm)

    # Truncation guard: if new train.py is cut off, regenerate it standalone
    if tm and not re.search(r'print\s*\(\s*json\.dumps', tm):
        import sys as _sys
        print("[engine] revise_after_iteration: train.py truncated (missing metrics print). Regenerating standalone.", file=_sys.stderr)
        fallback_code = ask(
            "Write a complete, concise Python ML training script. Must end with print(json.dumps(metrics)).",
            f"""Write the next experiment train.py based on this context.

ERROR FROM LAST RUN: {error or 'none'}
HISTORY: {hist_txt}

PROGRAM PLAN (follow this):
{program_md[:3000]}

Rules:
- Load data from the path defined in DATA_PATH at top of script
- Train/test split. Compute metrics on TEST set only.
- joblib.dump(model, 'model.pkl')
- Save predictions.png and residuals.png (wrap in try/except)
- FINAL LINE MUST BE: print(json.dumps({{"model":"Name","rmse":0.0,"r2":0.0,"train_rmse":0.0,"test_rmse":0.0,"train_r2":0.0,"test_r2":0.0}}))
- Keep under 150 lines — no Optuna if it makes the script too long

Output ONLY ```python block.""",
            6000
        )
        fb_clean, _ = apply_code_guardrails(extract_code(fallback_code))
        if fb_clean and re.search(r'print\s*\(\s*json\.dumps', fb_clean):
            tm = fb_clean

    return {
        "keep": keep,
        "new_program_md": pm.strip(),
        "new_train_py": tm.strip(),
        "reasoning": reasoning or ("KEEP" if keep else "DISCARD"),
    }

def init_results_tsv(ws):
    ws = pathlib.Path(ws)
    p = ws / "results.tsv"
    if not p.exists():
        p.write_text("experiment\tmodel\tmetric\tmetric_value\ttrain_rmse\ttest_rmse\ttrain_r2\ttest_r2\trmse\tmape\tmae\tr2\tnse\tstatus\tdescription\n")
    return p

def append_results_tsv(ws, exp_num, model, metric_name, metric_value, status, description, all_metrics=None):
    ws = pathlib.Path(ws)
    p = init_results_tsv(ws)
    am = all_metrics or {}
    def _g(k): return f"{am[k]:.4f}" if k in am else ""
    cols = [_g(k) for k in ("train_rmse", "test_rmse", "train_r2", "test_r2", "rmse", "mape", "mae", "r2", "nse")]
    line = f"{exp_num:02d}\t{model}\t{metric_name}\t{metric_value:.6f}\t" + "\t".join(cols) + f"\t{status}\t{description.replace(chr(9), ' ')[:220]}\n"
    with open(p, "a") as f:
        f.write(line)

def _nb_md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text],
    }

def _nb_code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [text],
    }

def write_prepare_py(ws, profile, obj, csv_path):
    ws = pathlib.Path(ws)
    code = f'''"""
Auto-generated data preparation script for this 19Labs run.
Usage:
    python prepare.py --input "{pathlib.Path(csv_path).name}" --target "{obj.get("target", "")}"
"""

import argparse
import json
from pathlib import Path

import pandas as pd

def run(input_csv: str, target: str, out_dir: str = "prepared"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if target and target not in df.columns:
        raise ValueError(f"Target '{{target}}' not found in columns: {{list(df.columns)}}")

    # Basic hygiene: remove duplicate rows and normalize missing values.
    df = df.drop_duplicates().replace([float("inf"), float("-inf")], pd.NA)

    dt_cols = []
    for c in df.columns:
        if df[c].dtype == "object":
            sample = df[c].dropna().astype(str).head(20)
            if sample.empty:
                continue
            if sample.str.contains(r"-|/").mean() >= 0.7:
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                    dt_cols.append(c)
                except Exception:
                    pass

    # Time-aware split when datetime exists; otherwise random split.
    if dt_cols:
        split_col = dt_cols[0]
        df = df.sort_values(split_col)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
    else:
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index).copy()

    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)

    summary = {{
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "target": target,
        "datetime_columns": dt_cols,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
    }}
    (out / "prep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="{pathlib.Path(csv_path).name}")
    ap.add_argument("--target", default="{obj.get("target", "")}")
    ap.add_argument("--out-dir", default="prepared")
    args = ap.parse_args()
    run(args.input, args.target, args.out_dir)
'''
    p = ws / "prepare.py"
    p.write_text(code)
    return str(p)

def write_analysis_notebook(ws, profile, obj):
    ws = pathlib.Path(ws)
    metric = obj.get("metric", "metric")
    direction = obj.get("direction", "lower_is_better")
    nb = {
        "cells": [
            _nb_md(
                "# 19Labs Project Analysis\n\n"
                "Auto-generated notebook for dataset profiling, experiment tracking, and progress visualization."
            ),
            _nb_md(
                f"## Project Context\n\n"
                f"- Task: **{obj.get('task', 'Unknown')}**\n"
                f"- Target: **{obj.get('target', 'target')}**\n"
                f"- Metric: **{metric}** ({direction})\n"
                f"- Rows x Cols: **{profile.get('rows', 0):,} x {profile.get('cols', 0)}**\n"
            ),
            _nb_code(
                "import pandas as pd\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n\n"
                "profile = pd.read_json('profile.json') if __import__('pathlib').Path('profile.json').exists() else None\n"
                "results = pd.read_csv('results.tsv', sep='\\t')\n"
                "results"
            ),
            _nb_code(
                f"metric_col = 'metric_value'\n"
                f"metric_name = '{metric}'\n"
                "df = pd.read_csv('results.tsv', sep='\\t')\n"
                "df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')\n"
                "df['experiment'] = pd.to_numeric(df['experiment'], errors='coerce')\n"
                "ok = df[df['status'] != 'crash'].copy()\n"
                "if len(ok) > 0:\n"
                "    if '" + direction + "' == 'lower_is_better':\n"
                "        ok['running_best'] = ok['metric_value'].cummin()\n"
                "    else:\n"
                "        ok['running_best'] = ok['metric_value'].cummax()\n"
                "    fig, ax = plt.subplots(figsize=(12, 5))\n"
                "    ax.plot(ok['experiment'], ok['metric_value'], marker='o', alpha=0.5, label='all')\n"
                "    ax.plot(ok['experiment'], ok['running_best'], marker='o', linewidth=2, label='running_best')\n"
                "    ax.set_title(f'Experiment Progress ({metric_name})')\n"
                "    ax.set_xlabel('Experiment')\n"
                "    ax.set_ylabel(metric_name)\n"
                "    ax.grid(alpha=0.2)\n"
                "    ax.legend()\n"
                "    plt.tight_layout()\n"
                "    plt.savefig('progress.png', dpi=150)\n"
                "    plt.show()\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    p = ws / "analysis.ipynb"
    p.write_text(json.dumps(nb, indent=2))
    return str(p)

def _apply_dark_style(plt):
    """Premium dark editorial style — high contrast, clean, publication-quality."""
    plt.rcParams.update({
        "figure.facecolor":       "#09090b",
        "axes.facecolor":         "#09090b",
        "axes.spines.top":        False,
        "axes.spines.right":      False,
        "axes.spines.left":       True,
        "axes.spines.bottom":     True,
        "axes.edgecolor":         "#27272a",
        "axes.labelcolor":        "#a1a1aa",
        "xtick.color":            "#52525b",
        "ytick.color":            "#71717a",
        "text.color":             "#e4e4e7",
        "axes.labelsize":         10,
        "xtick.labelsize":        9,
        "ytick.labelsize":        9,
        "xtick.major.size":       0,
        "ytick.major.size":       3,
        "xtick.major.pad":        6,
        "ytick.major.pad":        5,
        "axes.grid":              True,
        "axes.axisbelow":         True,
        "grid.color":             "#1c1c1f",
        "grid.linewidth":         0.8,
        "grid.alpha":             1.0,
        "xtick.bottom":           True,
        "ytick.left":             True,
        "legend.facecolor":       "#111113",
        "legend.edgecolor":       "#27272a",
        "legend.fontsize":        9,
        "legend.framealpha":      0.95,
        "legend.borderpad":       0.6,
        "legend.handlelength":    1.5,
        "font.family":            "sans-serif",
        "font.size":              10,
        "figure.dpi":             160,
    })


def render_progress_png(ws, history, obj):
    """Single progress chart updated after every experiment."""
    ws = pathlib.Path(ws)
    p  = ws / "progress.png"
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return ""

    _apply_dark_style(plt)
    BG     = "#09090b"
    BLUE   = "#3b82f6"
    PURPLE = "#8b5cf6"
    RED    = "#ef4444"
    MUTED  = "#3f3f46"

    rows = [h for h in history if h.get("success")]
    if not rows:
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.text(0.5, 0.5, "Waiting for first result…",
                ha="center", va="center", fontsize=13, color="#52525b")
        ax.set_facecolor(BG); fig.set_facecolor(BG); ax.axis("off")
        plt.savefig(p, dpi=160, facecolor=BG, bbox_inches="tight"); plt.close(fig)
        return str(p)

    metric = obj.get("metric", "metric")
    lower  = obj.get("direction", "lower_is_better") == "lower_is_better"
    xs     = [h["num"] for h in rows]
    ys     = [float(h["metric_val"]) for h in rows]
    train_key = f"train_{metric}"
    train_ys = [h.get("all_metrics", {}).get(train_key) for h in rows]
    has_train = any(v is not None for v in train_ys)

    running, cur = [], None
    for y in ys:
        cur = y if cur is None else (min(cur, y) if lower else max(cur, y))
        running.append(cur)

    bi = running.index(min(running) if lower else max(running))
    best_val = running[bi]
    best_model = rows[bi].get("model", "")

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Gradient fill under running best (manual polygon)
    ax.fill_between(xs, running, min(ys + running) - (max(ys + running) - min(ys + running)) * 0.2,
                    color=BLUE, alpha=0.07, zorder=1)

    # Train line (if available)
    if has_train:
        ty = [v if v is not None else float("nan") for v in train_ys]
        ax.plot(xs, ty, color=PURPLE, linewidth=1.2, linestyle="--",
                alpha=0.55, label="train", zorder=2, solid_capstyle="round")

    # Individual experiment dots (colored by status)
    discarded_xs = [xs[i] for i, h in enumerate(rows) if h.get("status") == "discard"]
    discarded_ys = [ys[i] for i, h in enumerate(rows) if h.get("status") == "discard"]
    kept_xs = [xs[i] for i, h in enumerate(rows) if h.get("status") != "discard"]
    kept_ys = [ys[i] for i, h in enumerate(rows) if h.get("status") != "discard"]

    if discarded_xs:
        ax.scatter(discarded_xs, discarded_ys, s=22, color=RED, zorder=3,
                   linewidths=0, alpha=0.4, label="discarded")
    ax.scatter(kept_xs, kept_ys, s=28, color=MUTED, zorder=3, linewidths=0, label="kept")

    # Running best line
    ax.plot(xs, running, color=BLUE, linewidth=2.5, zorder=4,
            solid_capstyle="round", label="best so far")

    # Best experiment highlight
    ax.scatter([xs[bi]], [best_val], s=80, color=BLUE, zorder=5, linewidths=0,
               edgecolors="#09090b")
    ax.scatter([xs[bi]], [best_val], s=200, color=BLUE, zorder=4, linewidths=0,
               alpha=0.15)

    # Annotate best value
    v_range = max(ys + running) - min(ys + running) or abs(best_val) * 0.1 or 0.01
    fmt = f"{best_val:.4f}" if abs(best_val) < 100 else f"{best_val:,.1f}"
    ax.annotate(
        f"  best: {fmt}\n  {best_model[:20]}",
        (xs[bi], best_val),
        xytext=(8, 12), textcoords="offset points",
        fontsize=9, color=BLUE, fontweight="600",
        va="bottom",
    )

    # Titles and axes
    ax.set_title(
        f"{metric.upper()}  ·  {len(rows)} experiments",
        fontsize=13, fontweight="700", color="#fafafa", loc="left", pad=14,
    )
    ax.set_xlabel("experiment #", labelpad=8)
    ax.spines["left"].set_color("#27272a")
    ax.spines["bottom"].set_color("#27272a")
    ax.tick_params(axis="x", length=0, pad=6, colors="#52525b")
    ax.tick_params(axis="y", length=3, pad=5, colors="#71717a")
    ax.yaxis.set_visible(True)

    if len(rows) <= 15:
        ax.set_xticks(xs)
        ax.set_xticklabels([f"#{x}" for x in xs], fontsize=8, color="#52525b")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout(pad=1.8)
    plt.savefig(p, dpi=180, facecolor=BG, bbox_inches="tight"); plt.close(fig)
    return str(p)


def render_final_plots(ws, history, obj, best):
    """Generate clean minimal final plots after all experiments."""
    ws = pathlib.Path(ws)
    generated = {}
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return generated

    _apply_dark_style(plt)
    rows  = [h for h in history if h.get("success")]
    if not rows:
        return generated

    BG           = "#09090b"
    BLUE, PURPLE = "#3b82f6", "#8b5cf6"
    RED, MUTED   = "#ef4444", "#3f3f46"
    DIM          = "#27272a"
    metric = obj.get("metric", "metric")
    lower  = obj.get("direction", "lower_is_better") == "lower_is_better"

    def _save(fig, name):
        plt.tight_layout(pad=2.0)
        p = ws / name
        plt.savefig(p, dpi=180, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
        return str(p)

    def _clean_ax(ax):
        ax.spines["bottom"].set_color(DIM)
        ax.tick_params(length=0, pad=6)

    # ── 1. EXPERIMENT TIMELINE ──────────────────────────────────────
    try:
        xs        = [h["num"] for h in rows]
        test_vals = [float(h["metric_val"]) for h in rows]
        train_key = f"train_{metric}"
        train_raw = [h.get("all_metrics", {}).get(train_key) for h in rows]
        has_train = any(v is not None for v in train_raw)

        running, cur = [], None
        for y in test_vals:
            cur = y if cur is None else (min(cur, y) if lower else max(cur, y))
            running.append(cur)
        bi = running.index(min(running) if lower else max(running))

        fig, ax = plt.subplots(figsize=(13, 5))

        # Fill under best line
        baseline = min(test_vals + running) - (max(test_vals + running) - min(test_vals + running)) * 0.15
        ax.fill_between(xs, running, baseline, color=BLUE, alpha=0.07, zorder=1)

        if has_train:
            tv = [v if v is not None else float("nan") for v in train_raw]
            ax.plot(xs, tv, color=PURPLE, linewidth=1.3, linestyle="--",
                    alpha=0.55, label="train", zorder=2, solid_capstyle="round")

        # Color dots: discarded=red, kept=dim, best=blue
        for i, h in enumerate(rows):
            is_best = (i == bi)
            is_disc = h.get("status") == "discard"
            col = BLUE if is_best else (RED if is_disc else MUTED)
            alpha = 1.0 if is_best else (0.4 if is_disc else 0.7)
            ax.scatter([xs[i]], [test_vals[i]], s=(80 if is_best else 28),
                       color=col, zorder=4, linewidths=0, alpha=alpha)

        ax.plot(xs, running, color=BLUE, linewidth=2.5, zorder=3,
                solid_capstyle="round", label="best so far")

        crashes = [h["num"] for h in history if not h.get("success")]
        if crashes:
            ylim_min = min(test_vals + running) - abs(max(test_vals + running) - min(test_vals + running)) * 0.12
            ax.scatter(crashes, [ylim_min] * len(crashes),
                       marker="|", color=RED, s=60, zorder=5, label=f"failed ({len(crashes)})", alpha=0.7)

        best_val = running[bi]
        best_model = rows[bi].get("model", "")
        fmt = f"{best_val:.4f}" if abs(best_val) < 100 else f"{best_val:,.1f}"
        ax.set_title(
            f"{metric.upper()}  ·  best: {fmt}  ({best_model})",
            fontsize=13, fontweight="700", color="#e4e4e7", loc="left", pad=16,
        )
        ax.set_xlabel("experiment #", labelpad=8)
        ax.spines["left"].set_color("#27272a")
        ax.spines["bottom"].set_color("#27272a")
        ax.yaxis.set_visible(True)
        ax.tick_params(axis="y", colors="#71717a", length=3)
        ax.tick_params(axis="x", length=0, pad=6)
        if len(xs) <= 15:
            ax.set_xticks(xs)
            ax.set_xticklabels([f"#{x}" for x in xs], fontsize=8, color="#52525b")
        legend_items = ["train", "best so far"] if has_train else ["best so far"]
        if crashes:
            legend_items.append(f"failed ({len(crashes)})")
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout(pad=2.0)
        generated["experiment_timeline_png"] = _save(fig, "experiment_timeline.png")
    except Exception:
        pass

    # ── 2. MODEL COMPARISON ─────────────────────────────────────────
    try:
        from collections import OrderedDict
        model_best = OrderedDict()
        for h in rows:
            m = h.get("model", "?")
            v = float((h.get("all_metrics") or {}).get(f"test_{metric}") or h["metric_val"])
            if m not in model_best or (lower and v < model_best[m]) or (not lower and v > model_best[m]):
                model_best[m] = v

        if model_best:
            sorted_m = sorted(model_best.items(), key=lambda x: x[1], reverse=not lower)
            names  = [m[:32] for m, _ in sorted_m]
            vals   = [v for _, v in sorted_m]
            n_bars = len(names)

            # Color bars: best=blue, top3=indigo, rest=muted
            colors = []
            for i in range(n_bars):
                if i == 0: colors.append(BLUE)
                elif i <= 2: colors.append("#6366f1")
                else: colors.append(MUTED)

            fig, ax = plt.subplots(figsize=(10, max(3.0, n_bars * 0.6 + 1.5)))
            y_pos = np.arange(n_bars)
            bars = ax.barh(y_pos, vals, color=colors, height=0.5, zorder=3,
                           edgecolor="none")

            # Add a subtle highlight on the top half of each bar
            for i, bar in enumerate(bars):
                ax.barh(y_pos[i], bar.get_width(), height=0.25,
                        color="white", alpha=0.05, zorder=4, edgecolor="none",
                        align="edge")

            # Value labels on each bar
            for i, (name, v) in enumerate(zip(names, vals)):
                fmt = f"{v:.4f}" if abs(v) < 100 else f"{v:,.1f}"
                x_off = (max(vals) - min(vals)) * 0.01 if vals else 0
                ax.text(v + x_off, y_pos[i], fmt,
                        va="center", ha="left",
                        fontsize=9, fontweight="700" if i == 0 else "400",
                        color="#e4e4e7" if i == 0 else "#71717a")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=10,
                               color=["#e4e4e7" if i == 0 else "#a1a1aa" for i in range(n_bars)])
            ax.set_title(
                f"model comparison  ·  {metric.upper()}  ({'lower better' if lower else 'higher better'})",
                fontsize=13, fontweight="700", color="#e4e4e7", loc="left", pad=16,
            )
            ax.xaxis.set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_color("#27272a")
            ax.invert_yaxis()
            ax.set_xlim(right=max(vals) * (1.15 if not lower else 1.15))
            plt.tight_layout(pad=2.0)
            generated["model_comparison_png"] = _save(fig, "model_comparison.png")
    except Exception:
        pass

    # ── 3. METRICS SCORECARD ─────────────────────────────────────────
    try:
        best_am = (best or {}).get("all_metrics", {})
        pairs = []
        for base in ["rmse", "r2", "mape", "mae", "nse", "accuracy", "f1", "auc"]:
            tr = best_am.get(f"train_{base}")
            te = best_am.get(f"test_{base}") or best_am.get(base)
            if tr is not None and te is not None:
                pairs.append((base.upper(), float(tr), float(te)))

        if pairs:
            model_name = (best or {}).get("model", "Best Model")
            n = len(pairs)
            fig, axes = plt.subplots(1, n, figsize=(max(8, 2.8 * n), 4.0))
            if n == 1:
                axes = [axes]
            fig.suptitle(f"{model_name}  ·  train vs test metrics",
                         fontsize=12, color="#a1a1aa", fontweight="500",
                         y=1.04, x=0.0, ha="left")

            for i, (name, tr, te) in enumerate(pairs):
                ax = axes[i]
                overfit = abs(tr - te) / (max(abs(tr), abs(te), 1e-10)) > 0.2
                tr_col = "#f59e0b" if overfit else DIM
                te_col = BLUE

                # Fill between
                ax.fill_between([0, 1], [tr, te], alpha=0.07,
                                color=RED if overfit else BLUE)
                ax.plot([0, 1], [tr, te], color=MUTED, linewidth=1.2,
                        zorder=2, alpha=0.7)
                ax.scatter([0], [tr], s=140, color=tr_col, zorder=4,
                           linewidths=0, edgecolors="#09090b")
                ax.scatter([1], [te], s=140, color=te_col, zorder=4,
                           linewidths=0, edgecolors="#09090b")

                fmt_tr = f"{tr:.4f}" if abs(tr) < 10 else f"{tr:,.2f}"
                fmt_te = f"{te:.4f}" if abs(te) < 10 else f"{te:,.2f}"
                offset = (max(tr, te) - min(tr, te)) * 0.12 + 0.0001
                ax.text(0, tr + offset, fmt_tr, ha="center", va="bottom",
                        fontsize=9, color="#a1a1aa", fontweight="500")
                ax.text(1, te + offset, fmt_te, ha="center", va="bottom",
                        fontsize=10, color="#e4e4e7", fontweight="700")

                if overfit:
                    ax.text(0.5, (tr + te) / 2, "⚠", ha="center", va="center",
                            fontsize=11, color=RED, alpha=0.7)

                ax.set_xticks([0, 1])
                ax.set_xticklabels(["train", "test"], fontsize=10, color="#71717a")
                ax.set_title(name, fontsize=11, color="#a1a1aa", pad=12, fontweight="600")
                ax.yaxis.set_visible(False)
                ax.spines["bottom"].set_color("#27272a")
                ax.spines["left"].set_visible(False)
                ax.tick_params(length=0, pad=8)
                margin = (max(tr, te) - min(tr, te)) * 0.5 + 0.0001
                ax.set_ylim(min(tr, te) - margin * 1.5, max(tr, te) + margin * 3)
                ax.set_xlim(-0.4, 1.4)

            plt.tight_layout(pad=2.0)
            p = ws / "metrics_overview.png"
            plt.savefig(p, dpi=180, facecolor=BG, bbox_inches="tight")
            plt.close(fig)
            generated["metrics_overview_png"] = str(p)
    except Exception:
        pass

    # ── 4. TRAIN vs TEST per experiment ──────────────────────────────
    try:
        avail = [(tk, vk, lbl) for tk, vk, lbl in [
            ("train_rmse", "test_rmse", "RMSE"),
            ("train_r2",   "test_r2",   "R²"),
            ("train_mape", "test_mape", "MAPE"),
            ("train_mae",  "test_mae",  "MAE"),
            (f"train_{metric}", f"test_{metric}", metric.upper()),
        ] if any(h.get("all_metrics", {}).get(tk) is not None for h in rows)]

        # Remove duplicates
        seen_lbl = set()
        avail = [x for x in avail if not (x[2] in seen_lbl or seen_lbl.add(x[2]))]

        if avail and len(rows) >= 2:
            n_p = min(len(avail), 3)
            avail = avail[:n_p]
            fig, axes = plt.subplots(1, n_p, figsize=(5.5 * n_p, 4.5))
            if n_p == 1:
                axes = [axes]
            xs_idx = np.arange(len(rows))
            w = 0.35

            for i, (tk, vk, lbl) in enumerate(avail):
                ax = axes[i]
                trains = [h.get("all_metrics", {}).get(tk) or 0 for h in rows]
                tests  = [h.get("all_metrics", {}).get(vk) or 0 for h in rows]

                # Detect overfitting per experiment
                overfits = [abs(tr - te) / (max(abs(te), 1e-10)) > 0.25 for tr, te in zip(trains, tests)]
                bar_colors = [RED if ov else BLUE for ov in overfits]

                ax.bar(xs_idx - w/2, trains, w, color=MUTED, label="train", zorder=3,
                       edgecolor="none", alpha=0.85)
                for j, (xi, te, col) in enumerate(zip(xs_idx, tests, bar_colors)):
                    ax.bar(xi + w/2, te, w, color=col, zorder=3, edgecolor="none", alpha=0.9)

                ax.set_xticks(xs_idx)
                ax.set_xticklabels([f"#{h['num']}" for h in rows], fontsize=8, color="#52525b")
                ax.set_title(lbl, fontsize=12, color="#a1a1aa", pad=12, fontweight="600")
                ax.yaxis.set_visible(True)
                ax.tick_params(axis="y", colors="#71717a", length=3)
                ax.tick_params(axis="x", length=0, pad=6)
                ax.spines["left"].set_color("#27272a")
                ax.spines["bottom"].set_color("#27272a")
                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(color=MUTED, label="train"), Patch(color=BLUE, label="test")],
                          loc="upper right", fontsize=9)

            generated["train_test_png"] = _save(fig, "train_test.png")
    except Exception:
        pass

    # ── 5. DATA CORRELATION HEATMAP (from CSV) ───────────────────────
    if not (ws / "correlation.png").exists():
        try:
            import pandas as pd
            csv_files = list(ws.glob("*.csv")) + list(ws.glob("*.tsv"))
            if csv_files:
                df = pd.read_csv(csv_files[0], sep=None, engine="python", nrows=5000)
                num_cols = df.select_dtypes(include="number").columns.tolist()
                target = obj.get("target", "")
                if target in num_cols and len(num_cols) >= 3:
                    # Keep top-N features by abs correlation with target
                    corr_with_target = df[num_cols].corr()[target].abs().sort_values(ascending=False)
                    keep = corr_with_target.head(20).index.tolist()
                    corr = df[keep].corr()
                    n = len(keep)
                    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.55)))
                    try:
                        import seaborn as sns
                        sns.heatmap(
                            corr, annot=True, fmt=".2f", cmap="RdBu_r",
                            center=0, vmin=-1, vmax=1,
                            linewidths=0.4, linecolor="#18181b",
                            annot_kws={"size": 7, "color": "#fafafa"},
                            ax=ax, cbar_kws={"shrink": 0.7},
                        )
                        ax.tick_params(colors="#71717a", labelsize=8)
                        cbar = ax.collections[0].colorbar
                        cbar.ax.tick_params(colors="#52525b", labelsize=7)
                    except Exception:
                        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                        ax.set_xticks(range(n)); ax.set_xticklabels(keep, rotation=45, ha="right", fontsize=7, color="#71717a")
                        ax.set_yticks(range(n)); ax.set_yticklabels(keep, fontsize=7, color="#71717a")
                        for i in range(n):
                            for j in range(n):
                                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=6, color="#fafafa")
                    ax.set_title(
                        f"feature correlations  ·  top {n} by |r| with {target}",
                        fontsize=11, fontweight="600", color="#fafafa", loc="left", pad=14,
                    )
                    ax.set_facecolor(BG); fig.set_facecolor(BG)
                    plt.tight_layout(pad=1.6)
                    p = ws / "correlation.png"
                    plt.savefig(p, dpi=160, facecolor=BG, bbox_inches="tight")
                    plt.close(fig)
                    generated["correlation_png"] = str(p)
        except Exception:
            pass

    # ── 6. DATA TIME SERIES (from CSV, if not already generated by train.py) ──
    if not (ws / "timeseries.png").exists():
        try:
            import pandas as pd
            csv_files = list(ws.glob("*.csv")) + list(ws.glob("*.tsv"))
            # Exclude small files (results/metrics) — prefer the actual data file
            csv_files = sorted(csv_files, key=lambda f: -f.stat().st_size)
            if csv_files:
                df = pd.read_csv(csv_files[0], sep=None, engine="python", nrows=30000)
                # Detect date column
                date_col = None
                for col in df.columns:
                    if "date" in col.lower() or "time" in col.lower() or "period" in col.lower() or "month" in col.lower() or "year" in col.lower() or "week" in col.lower():
                        try:
                            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                            if parsed.notna().mean() > 0.7:
                                df[col] = parsed
                                date_col = col
                                break
                        except Exception:
                            pass
                if not date_col:
                    for col in df.columns:
                        if df[col].dtype == object:
                            try:
                                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                                if parsed.notna().mean() > 0.8:
                                    df[col] = parsed
                                    date_col = col
                                    break
                            except Exception:
                                pass
                target = obj.get("target", "")
                if date_col and target in df.columns:
                    df = df.sort_values(date_col).reset_index(drop=True)
                    split_idx = int(len(df) * 0.8)
                    dates = df[date_col]
                    actuals = pd.to_numeric(df[target], errors="coerce")

                    # Try to load predictions saved by train.py
                    pred_files = sorted(
                        list(ws.glob("pred*.csv")) + list(ws.glob("*pred*.csv")) +
                        list(ws.glob("*forecast*.csv")) + list(ws.glob("*output*.csv")),
                        key=lambda f: f.stat().st_mtime, reverse=True
                    )
                    pred_series = None
                    if pred_files:
                        try:
                            pf = pd.read_csv(pred_files[0])
                            # Look for predicted/forecast column
                            for col in ["predicted", "forecast", "y_pred", "prediction", "pred"]:
                                if col in pf.columns:
                                    pred_series = pd.to_numeric(pf[col], errors="coerce")
                                    break
                            if pred_series is None and len(pf.columns) == 1:
                                pred_series = pd.to_numeric(pf.iloc[:, 0], errors="coerce")
                        except Exception:
                            pass

                    GREEN = "#34d399"
                    fig, ax = plt.subplots(figsize=(15, 5))

                    # Shade train region
                    ax.axvspan(dates.iloc[0], dates.iloc[split_idx],
                               alpha=0.05, color=BLUE, zorder=0)
                    ax.axvline(x=dates.iloc[split_idx], color=DIM, lw=1.2,
                               linestyle=":", zorder=2, alpha=0.8)

                    # Actual values
                    ax.plot(dates, actuals,
                            color="#e4e4e7", lw=1.4, alpha=0.9, label="actual", zorder=3)

                    # Predicted overlay on test set
                    if pred_series is not None and len(pred_series) > 0:
                        test_dates = dates.iloc[split_idx:split_idx + len(pred_series)]
                        ax.plot(test_dates, pred_series.values[:len(test_dates)],
                                color=BLUE, lw=2.8, alpha=0.95,
                                label="predicted (test)", zorder=5, solid_capstyle="round")
                        # Shade between actual and predicted on test
                        min_len = min(len(test_dates), len(pred_series))
                        ax.fill_between(
                            test_dates[:min_len],
                            actuals.iloc[split_idx:split_idx + min_len].values,
                            pred_series.values[:min_len],
                            alpha=0.08, color=BLUE, zorder=4
                        )
                    else:
                        # No predictions available — just show test actual in blue
                        ax.plot(dates.iloc[split_idx:], actuals.iloc[split_idx:],
                                color=BLUE, lw=2.2, alpha=0.9, label="test", zorder=4)

                    # Train/test region labels
                    y_range = actuals.max() - actuals.min()
                    y_top = actuals.max() + y_range * 0.04
                    ax.text(dates.iloc[split_idx // 2], y_top, "TRAIN",
                            fontsize=8, color="#52525b", ha="center", va="bottom",
                            fontweight="600", alpha=0.7)
                    ax.text(dates.iloc[split_idx + (len(dates) - split_idx) // 2], y_top, "TEST",
                            fontsize=8, color="#52525b", ha="center", va="bottom",
                            fontweight="600", alpha=0.7)

                    ax.set_title(
                        f"{target}  ·  actual vs predicted  ·  train | test split",
                        fontsize=13, fontweight="700", color="#e4e4e7", loc="left", pad=16,
                    )
                    ax.set_xlabel(date_col, labelpad=8)
                    ax.spines["left"].set_color("#27272a")
                    ax.spines["bottom"].set_color("#27272a")
                    ax.tick_params(axis="x", length=0, pad=6, colors="#52525b")
                    ax.tick_params(axis="y", length=3, pad=5, colors="#71717a")
                    ax.yaxis.set_visible(True)
                    ax.legend(loc="upper left", fontsize=9, frameon=True)
                    ax.set_facecolor(BG); fig.set_facecolor(BG)
                    plt.tight_layout(pad=2.0)
                    p = ws / "timeseries.png"
                    plt.savefig(p, dpi=180, facecolor=BG, bbox_inches="tight")
                    plt.close(fig)
                    generated["timeseries_png"] = str(p)
        except Exception:
            pass

    # ── 7. DATA OVERVIEW — multi-line normalized time series ────────
    if not (ws / "data_overview.png").exists():
        try:
            import pandas as pd
            csv_files = sorted(
                list(ws.glob("*.csv")) + list(ws.glob("*.tsv")),
                key=lambda f: -f.stat().st_size
            )
            if csv_files:
                df = pd.read_csv(csv_files[0], sep=None, engine="python", nrows=30000)
                # Detect date column
                date_col = None
                for col in df.columns:
                    if any(k in col.lower() for k in ["date","time","period","month","year","week","day"]):
                        try:
                            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                            if parsed.notna().mean() > 0.7:
                                df[col] = parsed; date_col = col; break
                        except Exception: pass
                if not date_col:
                    for col in df.columns:
                        if df[col].dtype == object:
                            try:
                                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                                if parsed.notna().mean() > 0.8:
                                    df[col] = parsed; date_col = col; break
                            except Exception: pass

                target = obj.get("target", "")
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if date_col and len(num_cols) >= 2:
                    df = df.sort_values(date_col).reset_index(drop=True)
                    dates = df[date_col]
                    # Pick target + top correlated features (up to 5 total)
                    plot_cols = [c for c in num_cols if c != target]
                    if target in num_cols:
                        try:
                            corr_rank = df[num_cols].corr()[target].abs().drop(target).sort_values(ascending=False)
                            plot_cols = corr_rank.head(4).index.tolist()
                        except Exception:
                            plot_cols = plot_cols[:4]
                        plot_cols = [target] + plot_cols
                    else:
                        plot_cols = num_cols[:5]

                    palette = ["#e4e4e7", BLUE, "#34d399", "#f59e0b", PURPLE]
                    n_lines = min(len(plot_cols), 5)
                    plot_cols = plot_cols[:n_lines]

                    fig, axes = plt.subplots(n_lines, 1, figsize=(15, 2.5 * n_lines), sharex=True)
                    if n_lines == 1: axes = [axes]
                    fig.suptitle(
                        "Data Overview  ·  all key signals over time",
                        fontsize=13, fontweight="700", color="#e4e4e7", x=0.01, ha="left", y=1.01
                    )

                    for i, (col, color) in enumerate(zip(plot_cols, palette)):
                        ax = axes[i]
                        series = pd.to_numeric(df[col], errors="coerce")
                        # Rolling average
                        window = max(3, len(series) // 50)
                        rolling = series.rolling(window, center=True, min_periods=1).mean()

                        ax.fill_between(dates, series, series.min(), alpha=0.06, color=color, zorder=1)
                        ax.plot(dates, series, color=color, lw=1.0, alpha=0.45, zorder=2)
                        ax.plot(dates, rolling, color=color, lw=2.2, alpha=0.95, zorder=3,
                                solid_capstyle="round")

                        # Min/max annotation
                        ax.set_ylabel(col[:20], fontsize=9, color=color, labelpad=6)
                        ax.yaxis.label.set_rotation(0)
                        ax.yaxis.label.set_ha("right")
                        ax.tick_params(axis="y", colors="#52525b", length=2, labelsize=7)
                        ax.tick_params(axis="x", length=0, pad=6, colors="#52525b", labelsize=7)
                        ax.spines["left"].set_color(DIM)
                        ax.spines["bottom"].set_color(DIM)
                        ax.set_facecolor(BG)
                        # Mark overall max
                        try:
                            pk = series.idxmax()
                            ax.scatter([dates.iloc[pk]], [series.iloc[pk]], s=50, color=color,
                                       zorder=5, linewidths=0, alpha=0.9)
                            fmt_pk = f"{series.iloc[pk]:,.0f}" if series.iloc[pk] > 100 else f"{series.iloc[pk]:.3f}"
                            ax.annotate(fmt_pk, (dates.iloc[pk], series.iloc[pk]),
                                        xytext=(4, 6), textcoords="offset points",
                                        fontsize=7, color=color, alpha=0.8)
                        except Exception: pass
                        if i < n_lines - 1:
                            ax.spines["bottom"].set_visible(False)
                            ax.tick_params(axis="x", labelbottom=False)

                    axes[-1].set_xlabel(date_col, labelpad=6, fontsize=8, color="#52525b")
                    plt.tight_layout(pad=1.4)
                    p = ws / "data_overview.png"
                    plt.savefig(p, dpi=160, facecolor=BG, bbox_inches="tight")
                    plt.close(fig)
                    generated["data_overview_png"] = str(p)
        except Exception:
            pass

    # ── 8. SEASONALITY HEATMAP ───────────────────────────────────────
    if not (ws / "seasonality.png").exists():
        try:
            import pandas as pd
            csv_files = sorted(
                list(ws.glob("*.csv")) + list(ws.glob("*.tsv")),
                key=lambda f: -f.stat().st_size
            )
            if csv_files:
                df = pd.read_csv(csv_files[0], sep=None, engine="python", nrows=30000)
                target = obj.get("target", "")
                date_col = None
                for col in df.columns:
                    if any(k in col.lower() for k in ["date","time","period","month","year","week"]):
                        try:
                            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                            if parsed.notna().mean() > 0.7:
                                df[col] = parsed; date_col = col; break
                        except Exception: pass
                if not date_col:
                    for col in df.columns:
                        if df[col].dtype == object:
                            try:
                                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                                if parsed.notna().mean() > 0.8:
                                    df[col] = parsed; date_col = col; break
                            except Exception: pass

                if date_col and target in df.columns:
                    df = df.sort_values(date_col).reset_index(drop=True)
                    df["_target"] = pd.to_numeric(df[target], errors="coerce")
                    df["_month"]   = df[date_col].dt.month
                    df["_year"]    = df[date_col].dt.year
                    df["_dow"]     = df[date_col].dt.dayofweek  # 0=Mon

                    years  = df["_year"].nunique()
                    months = df["_month"].nunique()

                    # Decide chart type: year×month heatmap if multi-year, else dow×month
                    if years >= 2 and months >= 2:
                        pivot = df.groupby(["_year","_month"])["_target"].mean().unstack(fill_value=None)
                        row_lbl = [str(y) for y in pivot.index]
                        col_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                        col_lbl = col_lbl[:len(pivot.columns)]
                        title_sfx = "by year × month"
                    else:
                        dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                        pivot = df.groupby(["_dow","_month"])["_target"].mean().unstack(fill_value=None)
                        row_lbl = [dow_names[i] for i in pivot.index if i < len(dow_names)]
                        col_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                        col_lbl = col_lbl[:len(pivot.columns)]
                        title_sfx = "by weekday × month"

                    data = pivot.values.astype(float)
                    nr, nc = data.shape
                    fig, ax = plt.subplots(figsize=(max(8, nc * 0.9 + 2), max(4, nr * 0.55 + 2)))
                    import numpy as np
                    masked = np.ma.array(data, mask=np.isnan(data))
                    im = ax.imshow(masked, cmap="RdYlGn", aspect="auto",
                                   vmin=np.nanpercentile(data, 5),
                                   vmax=np.nanpercentile(data, 95))
                    ax.set_xticks(range(nc)); ax.set_xticklabels(col_lbl, fontsize=9, color="#a1a1aa")
                    ax.set_yticks(range(nr)); ax.set_yticklabels(row_lbl[:nr], fontsize=9, color="#a1a1aa")
                    # Value annotations
                    for i in range(nr):
                        for j in range(nc):
                            if not np.isnan(data[i, j]):
                                val = data[i, j]
                                fmt = f"{val:,.0f}" if abs(val) >= 100 else f"{val:.2f}"
                                ax.text(j, i, fmt, ha="center", va="center", fontsize=7,
                                        color="#fafafa", fontweight="500")
                    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                    cbar.ax.tick_params(colors="#52525b", labelsize=7)
                    ax.set_title(
                        f"Seasonality  ·  {target}  {title_sfx}",
                        fontsize=12, fontweight="700", color="#e4e4e7", loc="left", pad=14,
                    )
                    ax.set_facecolor(BG); fig.set_facecolor(BG)
                    plt.tight_layout(pad=1.6)
                    p = ws / "seasonality.png"
                    plt.savefig(p, dpi=160, facecolor=BG, bbox_inches="tight")
                    plt.close(fig)
                    generated["seasonality_png"] = str(p)
        except Exception:
            pass

    # ── 9. FEATURE IMPORTANCE (model-based or correlation-based) ─────
    if not (ws / "feature_importance.png").exists():
        try:
            import pandas as pd, numpy as np
            # Try to load from saved model
            importances, feat_names, method = None, None, "correlation"
            csv_files = sorted(list(ws.glob("*.csv")) + list(ws.glob("*.tsv")), key=lambda f: -f.stat().st_size)
            target = obj.get("target", "")

            # Method A: load from model.pkl
            for pkl_path in [ws / "best_model.pkl", ws / "model.pkl"]:
                if pkl_path.exists() and importances is None:
                    try:
                        import joblib
                        mdl = joblib.load(pkl_path)
                        # Unwrap pipeline
                        step = mdl
                        if hasattr(mdl, "steps"):
                            step = mdl.steps[-1][1]
                        if hasattr(step, "feature_importances_"):
                            importances = step.feature_importances_
                            method = "model"
                            # Try to get feature names from pipeline
                            if hasattr(mdl, "named_steps"):
                                for sname, sobj in reversed(mdl.steps):
                                    if hasattr(sobj, "get_feature_names_out"):
                                        try: feat_names = list(sobj.get_feature_names_out()); break
                                        except Exception: pass
                                    elif hasattr(sobj, "feature_names_in_"):
                                        try: feat_names = list(sobj.feature_names_in_); break
                                        except Exception: pass
                        elif hasattr(step, "coef_"):
                            importances = np.abs(step.coef_).flatten()
                            method = "model (coef)"
                    except Exception: pass

            # Method B: correlation with target from CSV
            if importances is None and csv_files and target:
                df = pd.read_csv(csv_files[0], sep=None, engine="python", nrows=10000)
                if target in df.columns:
                    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
                    if num_cols:
                        corr = df[num_cols].corrwith(pd.to_numeric(df[target], errors="coerce")).abs()
                        corr = corr.dropna().sort_values(ascending=False)
                        feat_names = list(corr.index[:20])
                        importances = corr.values[:20]
                        method = "|corr| with target"

            if importances is not None and len(importances) > 0:
                # If no feature names, generate generic ones
                if feat_names is None:
                    feat_names = [f"feature_{i}" for i in range(len(importances))]
                # Align lengths
                n = min(len(importances), len(feat_names), 20)
                imps = np.array(importances[:n])
                names = feat_names[:n]
                # Sort descending
                order = np.argsort(imps)[::-1]
                imps = imps[order]; names = [names[i] for i in order]
                # Truncate to top-20
                imps = imps[:20]; names = names[:20]
                n = len(imps)

                palette_imp = []
                for i in range(n):
                    t_pct = 1 - i / max(n - 1, 1)
                    palette_imp.append((
                        0.231 + (1 - t_pct) * (0.933 - 0.231),  # R
                        0.510 + (1 - t_pct) * (0.133 - 0.510),  # G
                        0.965 + (1 - t_pct) * (0.267 - 0.965),  # B
                    ))
                palette_imp = [(max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))
                               for r, g, b in palette_imp]

                fig, ax = plt.subplots(figsize=(10, max(4, n * 0.45 + 1.5)))
                y_pos = np.arange(n)
                ax.barh(y_pos, imps[::-1], color=palette_imp[::-1], height=0.6,
                        edgecolor="none", zorder=3)
                # Value labels
                for i, (v, nm) in enumerate(zip(imps[::-1], names[::-1])):
                    fmt = f"{v:.4f}" if v < 1 else f"{v:,.1f}"
                    ax.text(v + imps.max() * 0.01, i, fmt, va="center",
                            fontsize=8, color="#a1a1aa", fontweight="500" if i == n-1 else "400")
                ax.set_yticks(y_pos)
                ax.set_yticklabels([nm[:35] for nm in names[::-1]], fontsize=9,
                                   color=["#e4e4e7" if i == n-1 else "#71717a" for i in range(n)])
                ax.set_title(
                    f"Feature Importance  ·  {method}  ·  top {n} features",
                    fontsize=12, fontweight="700", color="#e4e4e7", loc="left", pad=14,
                )
                ax.xaxis.set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_color(DIM)
                ax.set_facecolor(BG); fig.set_facecolor(BG)
                plt.tight_layout(pad=2.0)
                p = ws / "feature_importance.png"
                plt.savefig(p, dpi=160, facecolor=BG, bbox_inches="tight")
                plt.close(fig)
                generated["feature_importance_png"] = str(p)
        except Exception:
            pass

    # ── 10. LAG CROSS-CORRELATION (causality signals) ───────────────
    if not (ws / "lag_correlation.png").exists():
        try:
            import pandas as pd, numpy as np
            csv_files = sorted(list(ws.glob("*.csv")) + list(ws.glob("*.tsv")), key=lambda f: -f.stat().st_size)
            target = obj.get("target", "")
            if csv_files and target:
                df = pd.read_csv(csv_files[0], sep=None, engine="python", nrows=10000)
                # Detect date col for ordering
                date_col = None
                for col in df.columns:
                    if any(k in col.lower() for k in ["date","time","period","month","year","week"]):
                        try:
                            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                            if parsed.notna().mean() > 0.7:
                                df[col] = parsed; date_col = col; break
                        except Exception: pass
                if date_col:
                    df = df.sort_values(date_col).reset_index(drop=True)

                if target in df.columns:
                    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
                    y = pd.to_numeric(df[target], errors="coerce").fillna(method="ffill").fillna(0)
                    y_std = (y - y.mean()) / (y.std() + 1e-10)

                    # Pick top features by instantaneous correlation
                    base_corrs = {}
                    for c in num_cols:
                        x = pd.to_numeric(df[c], errors="coerce").fillna(method="ffill").fillna(0)
                        try: base_corrs[c] = abs(np.corrcoef(x, y)[0, 1])
                        except Exception: base_corrs[c] = 0
                    top_feats = sorted(base_corrs, key=lambda k: -base_corrs[k])[:5]

                    if top_feats:
                        max_lag = min(20, len(df) // 5)
                        lags = range(-max_lag, max_lag + 1)
                        lag_arr = np.array(list(lags))

                        palette_lag = [BLUE, "#34d399", "#f59e0b", PURPLE, RED]
                        fig, ax = plt.subplots(figsize=(14, 5))

                        for feat, color in zip(top_feats, palette_lag):
                            x = pd.to_numeric(df[feat], errors="coerce").fillna(method="ffill").fillna(0)
                            x_std = (x - x.mean()) / (x.std() + 1e-10)
                            xcorr = []
                            for lag in lags:
                                if lag >= 0:
                                    a = x_std.iloc[:len(x_std) - lag] if lag > 0 else x_std
                                    b = y_std.iloc[lag:] if lag > 0 else y_std
                                else:
                                    a = x_std.iloc[-lag:]
                                    b = y_std.iloc[:len(y_std) + lag]
                                try: xcorr.append(float(np.corrcoef(a, b)[0, 1]))
                                except Exception: xcorr.append(0)
                            xcorr = np.array(xcorr)
                            ax.plot(lag_arr, xcorr, color=color, lw=2.0, alpha=0.85,
                                    label=feat[:25], solid_capstyle="round")
                            # Mark peak lag
                            pk_idx = int(np.argmax(np.abs(xcorr)))
                            ax.scatter([lag_arr[pk_idx]], [xcorr[pk_idx]], s=60, color=color,
                                       zorder=5, linewidths=0)

                        ax.axvline(0, color=DIM, lw=1.5, linestyle="--", alpha=0.7, label="lag=0")
                        ax.axhline(0, color=DIM, lw=0.8, alpha=0.4)
                        ax.fill_between(lag_arr[lag_arr < 0], -1, 1, alpha=0.03, color="#f59e0b")
                        ax.fill_between(lag_arr[lag_arr > 0], -1, 1, alpha=0.03, color=BLUE)

                        ax.text(-(max_lag * 0.6), 0.92, "features LEAD", fontsize=8,
                                color="#f59e0b", alpha=0.7, transform=ax.transData)
                        ax.text(max_lag * 0.25, 0.92, "features LAG", fontsize=8,
                                color=BLUE, alpha=0.7, transform=ax.transData)

                        ax.set_xlim(-max_lag, max_lag)
                        ax.set_ylim(-1.05, 1.1)
                        ax.set_xlabel("lag (periods)", labelpad=8, fontsize=9, color="#52525b")
                        ax.set_ylabel("cross-correlation", labelpad=8, fontsize=9, color="#52525b")
                        ax.set_title(
                            f"Lag Cross-Correlation  ·  how features relate to {target} over time",
                            fontsize=12, fontweight="700", color="#e4e4e7", loc="left", pad=14,
                        )
                        ax.spines["left"].set_color(DIM)
                        ax.spines["bottom"].set_color(DIM)
                        ax.tick_params(axis="both", colors="#52525b", length=3, labelsize=8)
                        ax.legend(loc="lower right", fontsize=8, frameon=True,
                                  facecolor="#18181b", edgecolor=DIM, labelcolor="#a1a1aa")
                        ax.set_facecolor(BG); fig.set_facecolor(BG)
                        plt.tight_layout(pad=2.0)
                        p = ws / "lag_correlation.png"
                        plt.savefig(p, dpi=160, facecolor=BG, bbox_inches="tight")
                        plt.close(fig)
                        generated["lag_correlation_png"] = str(p)
        except Exception:
            pass

    # Register any plots already saved by train.py that aren't in generated yet
    for key, fname in [
        ("timeseries_png","timeseries.png"), ("correlation_png","correlation.png"),
        ("shap_png","shap.png"), ("predictions_png","predictions.png"),
        ("residuals_png","residuals.png"), ("data_overview_png","data_overview.png"),
        ("seasonality_png","seasonality.png"), ("feature_importance_png","feature_importance.png"),
        ("lag_correlation_png","lag_correlation.png"),
    ]:
        if key not in generated and (ws / fname).exists():
            generated[key] = str(ws / fname)

    return generated

def initialize_workspace_artifacts(ws, csv_path, profile, obj):
    ws = pathlib.Path(ws)
    artifacts = {}
    artifacts["prepare_py"] = write_prepare_py(ws, profile, obj, csv_path)
    artifacts["analysis_ipynb"] = write_analysis_notebook(ws, profile, obj)
    artifacts["progress_png"] = render_progress_png(ws, [], obj)
    if AUTORESEARCH_DIR.exists():
        for name in ["program.md", "analysis.ipynb", "prepare.py"]:
            src = AUTORESEARCH_DIR / name
            if src.exists():
                dst = ws / f"template_{name}"
                dst.write_text(src.read_text())
                artifacts[f"template_{name.replace('.', '_')}"] = str(dst)
    return artifacts

# ── PACKAGE DEPLOYMENT ─────────────────────────────────────────
def package_deployment(ws, best_exp, obj, profile):
    ws = pathlib.Path(ws)
    d = ws / "deploy"; d.mkdir(exist_ok=True)
    src = ws / f"exp_{best_exp['num']:02d}.py"
    if src.exists():
        shutil.copy(src, d / "train.py")
    elif (ws / "train.py").exists():
        shutil.copy(ws / "train.py", d / "train.py")

    # Use best_model.pkl (preserved at each new-best event) — fall back to model.pkl
    model_src = None
    for candidate in [ws / "best_model.pkl", ws / "model.pkl"]:
        if candidate.exists():
            model_src = candidate
            break
    if model_src:
        shutil.copy2(model_src, d / "model.pkl")

    train_code = (d / "train.py").read_text() if (d / "train.py").exists() else ""

    inference = extract_code(ask(
        "Write production Python inference code. FastAPI endpoint + CLI. Output ONLY ```python.",
        f"""Production inference script for a trained ML model.

MODEL: {best_exp['model']} | TASK: {obj['task']} | TARGET: {obj['target']}
METRIC: {best_exp['metric_name']}={best_exp['metric_val']:.6f}
COLS: {', '.join(profile['headers'])}
NUMERIC: {', '.join(profile['numeric'])}
CATEGORICAL: {', '.join(profile['categorical'])}

TRAINING CODE (for reference on preprocessing):
```python
{train_code[:2000]}
```

Requirements:
- File is named inference_server.py, FastAPI app object MUST be named `app`
- Startup: load model.pkl via joblib (scan directory for *.pkl, *.joblib if not found)
- POST /predict — accepts flat JSON dict of feature values → returns {{"prediction": value}}
- GET /health — returns {{"status": "ok", "model": "{best_exp['model']}", "task": "{obj['task']}"}}
- Use $PORT env var (default 8000) for Railway compatibility
- Apply same preprocessing as training (encode categoricals, impute nulls)
- Handle missing/extra input fields gracefully

Output ONLY ```python code.""", 3000))

    (d / "inference_server.py").write_text(inference)
    dep_reqs = build_requirements_from_code(
        train_code,
        inference,
        extra_modules={"fastapi", "uvicorn", "joblib"},
    )
    (d / "requirements.txt").write_text(dep_reqs)

    # Railway + Docker deployment files
    (d / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "WORKDIR /app\n"
        "COPY . .\n"
        "RUN pip install --no-cache-dir -r requirements.txt\n"
        'CMD uvicorn inference_server:app --host 0.0.0.0 --port ${PORT:-8000}\n'
    )
    import json as _json
    (d / "railway.json").write_text(_json.dumps({
        "$schema": "https://railway.app/railway.schema.json",
        "deploy": {
            "startCommand": "uvicorn inference_server:app --host 0.0.0.0 --port $PORT",
            "healthcheckPath": "/health"
        }
    }, indent=2))

    history_lines = "\n".join(
        f"- Exp {h['num']:02d}: {h['model']} → {h['metric_name']}={h['metric_val']:.6f}"
        for h in best_exp.get("all_history", []))
    model_note = "✅ model.pkl included" if model_src else "⚠️  model.pkl not found — run train.py to generate it"
    (d / "README.md").write_text(f"""# 19Labs Deployed Model

## Model
- Task: {obj['task']}
- Target: `{obj['target']}`
- Model: **{best_exp['model']}**
- {best_exp['metric_name'].upper()}: **{best_exp['metric_val']:.6f}**
- Trained: {time.strftime('%Y-%m-%d %H:%M:%S')}
- {model_note}

## Deploy to Railway (1 command)
```bash
railway login && railway up
```

## Run locally
```bash
pip install -r requirements.txt
uvicorn inference_server:app --reload   # API on :8000
```

## Retrain
```bash
python train.py   # → saves model.pkl
```

## API
```bash
POST /predict
{{"feature1": 1.0, "feature2": "category"}}
→ {{"prediction": 42.0}}

GET /health
→ {{"status": "ok", "model": "{best_exp['model']}", "task": "{obj['task']}"}}
```

## Experiment History
{history_lines}
""")
    return str(d)

# ── REPORT ─────────────────────────────────────────────────────
def generate_report(ws, obj, profile, history, best):
    ws = pathlib.Path(ws)
    hist_txt = "\n".join(
        f"  Exp {h['num']:02d}: {h['model']:25s} {h['metric_name']}={h['metric_val']:.6f}"
        + (f" r2={h['r2']:.4f}" if h.get("r2") else "")
        + (" [FAILED]" if not h["success"] else "")
        for h in history)
    rpt = ask("Write precise technical ML research reports. Dense, no fluff.",
        f"""Final research report.

DOMAIN: {obj['domain']} | TASK: {obj['task']} | TARGET: {obj['target']}
METRIC: {obj['metric']} ({obj['direction']}) | GOOD ENOUGH: {obj.get('good_enough','')}
DATASET: {profile['rows']:,} rows × {profile['cols']} cols

EXPERIMENTS:
{hist_txt}

WINNER: {best['model']} {best['metric_name']}={best['metric_val']:.6f}

Sections: Executive Summary, Dataset, Experiment Breakdown, 
Winner Analysis, Production Notes, Next Steps. Technical, dense.""", 2000)
    (ws / "final_report.md").write_text(f"# 19Labs Research Report\n\n{rpt}")
    return rpt

def _grade_from_score(score: int) -> str:
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    return "D"

def build_run_diagnostics(obj, profile, history, best, deploy_path):
    total = len(history or [])
    success_rows = [h for h in (history or []) if h.get("success")]
    failures = [h for h in (history or []) if not h.get("success")]
    success_count = len(success_rows)
    fail_count = len(failures)
    success_rate = (success_count / total) if total else 0.0

    metric_name = (best or {}).get("metric_name") or obj.get("metric", "metric")
    metric_val = (best or {}).get("metric_val")
    direction = obj.get("direction", "lower_is_better")
    threshold = GOOD_ENOUGH.get(metric_name) or GOOD_ENOUGH.get(metric_name.lower())
    # Try to extract numeric threshold from Claude's good_enough string (e.g. "RMSE < 5,000")
    ge_str = obj.get("good_enough", "")
    ge_nums = re.findall(r'[\d,]+\.?\d*', ge_str.replace(',', ''))
    if ge_nums:
        try:
            parsed_thresh = float(ge_nums[0])
            if parsed_thresh > 0:
                threshold = parsed_thresh
        except ValueError:
            pass
    threshold_met = False
    if metric_val is not None and threshold is not None:
        threshold_met = (metric_val <= threshold) if direction == "lower_is_better" else (metric_val >= threshold)

    fail_reason_counts = {}
    for h in failures:
        fr = h.get("failure_reason") or classify_failure_reason(h.get("error", ""))
        fail_reason_counts[fr] = fail_reason_counts.get(fr, 0) + 1

    score = 30
    score += int(min(25, round(success_rate * 25)))
    if best:
        score += 15
        # Bonus for strong secondary metrics
        am = (best or {}).get("all_metrics", {})
        r2_val = am.get("r2") or am.get("R2")
        if r2_val is not None and r2_val > 0.85:
            score += 5
        if r2_val is not None and r2_val > 0.95:
            score += 5
    if threshold_met:
        score += 15
    if deploy_path:
        score += 10
    score = max(0, min(100, score))

    top_fail = sorted(fail_reason_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    risk_lines = [f"{k} ({v})" for k, v in top_fail] if top_fail else ["No dominant failure mode"]
    best_line = f"{best['model']} {metric_name}={float(metric_val):.6f}" if best and metric_val is not None else "No successful model yet"
    headline = (
        "Production-candidate run with deployable artifact."
        if best and deploy_path
        else "Exploratory run; reliability improvements still needed."
    )
    brief = (
        f"Dataset: {profile.get('rows', 0):,} rows x {profile.get('cols', 0)} cols. "
        f"Task: {obj.get('task', 'Unknown')} predicting {obj.get('target', 'target')}. "
        f"Best result: {best_line}. Success rate: {success_count}/{total}. "
        f"Top risks: {', '.join(risk_lines)}."
    )
    next_actions = [
        "Lock feature schema and add dataset contract checks before training.",
        "Run 3 repeated seeds for stability and report variance bands.",
        "Add offline backtest slice (time/segment) before shipping.",
    ]
    if threshold_met:
        next_actions[0] = "Promote current model to staged deployment and monitor drift."

    return {
        "yc_readiness_score": score,
        "yc_grade": _grade_from_score(score),
        "headline": headline,
        "executive_brief": brief,
        "success_rate": success_rate,
        "success_count": success_count,
        "failure_count": fail_count,
        "total_experiments": total,
        "best_metric_name": metric_name,
        "best_metric_value": metric_val,
        "threshold": threshold,
        "threshold_met": threshold_met,
        "top_failure_modes": risk_lines,
        "next_actions": next_actions,
        "reliability_mode": obj.get("reliability_mode", "balanced"),
    }

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run_research(
    csv_path,
    workspace=None,
    budget=6,
    user_hint="",
    api_key=None,
    log_callback=None,
    reliability_mode="balanced",
    continuous=False,
    cancel_event=None,
    provider="claude",
    model=None,
):
    """
    Karpathy-discipline autoresearch loop.
    
    Protocol:
    1. Git-init the workspace. Every experiment is a commit.
    2. KEEP = commit advances HEAD. DISCARD = git reset --hard HEAD~1.
    3. Only train.py is mutable by the AI. program.md is the research spec.
    4. prepare.py is a READ-ONLY utility template.
    5. All experiment output goes to run.log (never floods context).
    6. TIME_BUDGET is injected into every script for wall-clock discipline.
    7. In continuous mode: NEVER STOP. Stagnation triggers strategy pivot, not exit.
    8. Crash recovery: trivial fix → rerun, fundamental → skip and move on.
    """
    global AVAILABLE_PKGS
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if not key: raise RuntimeError("No API key.")
    _init_client(key, provider, model=model)
    reset_token_usage()

    mode = normalize_reliability_mode(reliability_mode)
    mode_cfg = RELIABILITY_PROFILES[mode]
    requested_budget = max(1, int(budget))

    if continuous:
        budget = int(mode_cfg.get("continuous_cap", MAX_CONTINUOUS_EXPERIMENTS))
        stagnation_limit = None  # NEVER STOP on stagnation in continuous mode
    else:
        budget = requested_budget  # User controls experiment count
        stagnation_limit = int(mode_cfg["stagnation_limit"])

    ws = pathlib.Path(workspace or tempfile.mkdtemp(prefix="19labs_"))
    ws.mkdir(parents=True, exist_ok=True)
    log = Logger(ws, log_callback)
    run_tag = f"run_{int(time.time())}"

    model_label = OPENAI_MODEL if _provider == "openai" else CLAUDE_MODEL
    log.sys(f"Workspace: {ws}")
    log.sys(f"Provider: {_provider} ({model_label})")
    log.sys(f"Budget: {budget} {'(continuous — NEVER STOP)' if continuous else f'(requested={requested_budget})'} | CSV: {csv_path}")
    log.sys(f"Reliability mode: {mode} | Policy: {mode_cfg['policy']}")
    log.sys(f"Time budget per experiment: {TIME_BUDGET}s | Hard kill: {EXEC_TIMEOUT}s")

    # ── GIT INIT ──────────────────────────────────────────────
    git_ok = git_init_workspace(ws, run_tag)
    if git_ok:
        log.engine("Git repo initialized — every experiment = 1 commit, DISCARD = hard reset")
    else:
        log.engine("Warning: git init failed, falling back to file-based tracking")

    # PROFILE
    log.engine("Profiling dataset...")
    narrate(log_callback, "profiling_start")
    _csv_path_obj = pathlib.Path(csv_path)
    if _csv_path_obj.is_dir():
        log.engine(f"Media dataset detected at {csv_path} — using media profiler")
        profile = profile_media_dataset(csv_path)
        log.engine(f"Media: {profile['media_type']} | {profile['total_files']} files | {profile['num_classes']} classes: {list(profile['classes'].keys())[:8]}")
    else:
        profile = profile_dataset(csv_path)
    log.engine(f"{profile['rows']:,} rows × {profile['cols']} cols | numeric={profile['numeric']} | cat={profile['categorical']}")
    log.engine(f"Signals: {' | '.join(profile.get('signals', []))}")
    narrate(log_callback, "profiling_done", rows=profile['rows'], cols=profile['cols'], signals='; '.join(profile.get('signals', [])[:2]))
    (ws / "profile.json").write_text(json.dumps(profile, indent=2, default=str))

    # DOMAIN INTELLIGENCE — reason like a senior data scientist
    log.engine("Analyzing domain and data quality...")
    narrate(log_callback, "domain_analysis")
    domain_analysis = analyze_domain(profile, user_hint)
    (ws / "domain_analysis.md").write_text(domain_analysis)
    # Log key lines from domain analysis
    domain_name = ""
    strategy_name = ""
    for line in domain_analysis.split("\n"):
        line = line.strip()
        if line and any(line.startswith(k) for k in ("INDUSTRY:", "PROBLEM_TYPE:", "MODELING_STRATEGY:", "CRITICAL_WARNINGS:")):
            log.claude(line[:200])
            if line.startswith("INDUSTRY:"):
                domain_name = line[len("INDUSTRY:"):].strip()[:80]
            if line.startswith("MODELING_STRATEGY:"):
                strategy_name = line[len("MODELING_STRATEGY:"):].strip()[:80]
    narrate(log_callback, "domain_done", domain=domain_name or "General", strategy=strategy_name or "standard ML pipeline")

    # INFER
    log.engine("Inferring task from data...")
    obj = infer_objective(profile, user_hint, domain_analysis=domain_analysis)
    log.claude(f"Task: {obj['task']} | Target: {obj['target']} | Metric: {obj['metric']} ({obj['direction']})")
    log.claude(f"Domain: {obj['domain']} | Confidence: {obj['confidence']:.0%}")
    log.claude(f"Good enough: {obj['good_enough']}")
    log.claude(obj["reasoning"])

    # Validate target column exists in the dataset
    if obj["target"] and obj["target"] not in profile["headers"]:
        close_match = [h for h in profile["headers"] if obj["target"].lower() in h.lower() or h.lower() in obj["target"].lower()]
        if close_match:
            log.engine(f"Target '{obj['target']}' not found in columns — correcting to '{close_match[0]}'")
            obj["target"] = close_match[0]
        else:
            fallback = profile["target_candidates"][0] if profile.get("target_candidates") else profile["headers"][-1]
            log.engine(f"Target '{obj['target']}' not found in columns — falling back to '{fallback}'")
            obj["target"] = fallback

    (ws / "objective.md").write_text(obj["raw"])
    (ws / "objective.json").write_text(json.dumps(obj, indent=2))
    artifacts = initialize_workspace_artifacts(ws, csv_path, profile, obj)
    log.engine("Initialized project artifacts (prepare.py, analysis.ipynb, progress.png)")

    # ── Detect Kaggle / competition context ──────────────────────────────
    _hint_lower = (user_hint or "").lower()
    _is_kaggle = "kaggle" in _hint_lower or "competition" in _hint_lower
    # Also auto-detect from workspace files: train + test + sample_submission = Kaggle pattern
    _ws = pathlib.Path(ws)
    _ws_csv_names = {p.name.lower() for p in _ws.glob("*.csv")}
    if not _is_kaggle and "test.csv" in _ws_csv_names and any(n.startswith("sample") for n in _ws_csv_names):
        _is_kaggle = True
    _kaggle_files = [p.name for p in _ws.glob("*.csv") if p.name.lower() != pathlib.Path(csv_path).name.lower()]
    _kaggle_test_file = next((n for n in _kaggle_files if "test" in n.lower()), None)
    _kaggle_sample_file = next((n for n in _kaggle_files if "sample" in n.lower() or "submission" in n.lower()), None)
    if _is_kaggle:
        log.engine(f"Kaggle competition detected — test={_kaggle_test_file}, sample={_kaggle_sample_file}")

    # ── INIT: program.md (mutable spec) + train.py (the ONLY file AI edits) ──
    history = []
    insights = []
    best = None
    best_train_py = ""
    best_val = None
    lower = obj["direction"] == "lower_is_better"
    obj["user_hint"] = user_hint
    obj["reliability_mode"] = mode
    obj["execution_policy"] = mode_cfg["policy"]
    obj["time_budget"] = TIME_BUDGET
    obj["is_kaggle"] = _is_kaggle
    obj["kaggle_test_file"] = _kaggle_test_file
    obj["kaggle_sample_file"] = _kaggle_sample_file
    no_improve_rounds = 0
    consecutive_crashes = 0

    init_results_tsv(ws)
    narrate(log_callback, "writing_plan")
    program_md = write_program_md(profile, obj, history, insights, domain_analysis=domain_analysis)
    (ws / "program.md").write_text(program_md)
    log.engine("program.md written (research spec — drives all experiments)")

    narrate(log_callback, "writing_code", num=1, approach="initial baseline model")
    train_py = write_train_py(program_md, profile, obj, 1, history, domain_analysis=domain_analysis)
    train_py, initial_notes = apply_code_guardrails(train_py)
    if initial_notes:
        log.engine(f"Applied train.py guardrails: {', '.join(sorted(set(initial_notes)))}")
    (ws / "train.py").write_text(f"DATA_PATH = {repr(str(csv_path))}\nDATA_SEP = {repr(profile.get('detected_sep', ','))}\nTIME_BUDGET = {TIME_BUDGET}\n\n{train_py}")
    write_workspace_requirements(ws, train_py)
    log.engine("train.py written from program.md")

    # Commit initial state
    if git_ok:
        git_commit_experiment(ws, 0, "initial: program.md + train.py from data profile")
        log.engine(f"Git commit: initial state @ {git_get_commit_hash(ws)}")

    # ══════════════════════════════════════════════════════════
    # THE LOOP — Karpathy-style: iterate, KEEP or DISCARD, never stop (continuous)
    # ══════════════════════════════════════════════════════════
    for n in range(1, budget + 1):
        if cancel_event and cancel_event.is_set():
            log.engine("Graceful stop — wrapping up with results from completed experiments.")
            break
        log.engine(f"\n{'═'*50}\nEXPERIMENT {n}/{budget} {'(continuous)' if continuous else ''}\n{'═'*50}")

        # Pre-exec guardrails
        train_py, pre_notes = apply_code_guardrails(train_py)
        if pre_notes:
            log.engine(f"Pre-exec guardrails: {', '.join(sorted(set(pre_notes)))}")

        # Auto-install any packages the script imports that aren't yet available
        installed = auto_install_packages(train_py, log)
        if installed:
            log.engine(f"Auto-installed: {', '.join(installed)}")
            narrate(log_callback, "installing", pkg=', '.join(installed))

        # RUN — all output to run.log, grep metrics from log
        narrate(log_callback, "executing", num=n, model=_infer_model_name(train_py))
        res = execute(train_py, csv_path, ws, n, data_sep=profile.get("detected_sep", ","))

        score = None
        error = None
        failure_reason = ""
        metric_name = obj["metric"]
        metric_val = 999 if lower else 0
        model_name = "Failed"
        what_worked = ""
        all_metrics = {}

        # Keys that look like metrics but aren't (row IDs, indices, counts)
        _NON_METRIC_KEYS = {"id", "row_id", "index", "count", "n", "epoch", "step",
                            "batch", "iter", "iteration", "size", "total", "num"}

        def _is_valid_metric(k, v):
            """Return True if (k, v) looks like a real ML metric."""
            if not isinstance(v, (int, float)):
                return False
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                return False
            if k.lower() in _NON_METRIC_KEYS:
                return False
            if k.lower() == "model":
                return False
            return True

        if res["success"]:
            consecutive_crashes = 0
            m = res["metrics"]
            score = m
            # Priority: exact primary key → test_ prefixed → uppercase → then fallbacks
            _primary = obj["metric"]  # what the user/system actually wants
            _search_order = [
                _primary, _primary.lower(), _primary.upper(),
                f"test_{_primary}", f"test_{_primary.lower()}", f"test_{_primary.upper()}",
            ]
            # Only fall back to other metrics if the primary is truly absent
            _fallbacks = [k for k in ["rmse", "r2", "mape", "mae", "nse", "auc", "f1", "accuracy"]
                          if k != _primary.lower()]
            _search_order += _fallbacks
            for k in _search_order:
                if k in m and _is_valid_metric(k, m[k]):
                    metric_val = float(m[k])
                    # Keep the canonical metric name (what user asked for) when it matches
                    if k.lower().replace("test_", "") == _primary.lower():
                        metric_name = _primary
                    else:
                        metric_name = k
                    break
            else:
                # Fallback: pick first key that looks like a real metric
                for k, v in m.items():
                    if _is_valid_metric(k, v):
                        metric_name = k
                        metric_val = float(v)
                        break
            # Collect ALL numeric metrics for multi-metric display
            all_metrics = {k: float(v) for k, v in m.items() if _is_valid_metric(k, v)}
            model_name = m.get("model") or _infer_model_name(train_py)
            what_worked = m.get("what_worked", "")

            # Detect sentinel/garbage metrics (code caught its own error and printed a fallback)
            stdout_text = res.get("stdout", "")
            has_error_in_log = bool(re.search(
                r'(?:FileNotFoundError|KeyError|ValueError|ModuleNotFoundError|Exception|Error|Traceback)',
                stdout_text[-1000:], re.IGNORECASE
            ))
            is_nan_or_inf = math.isnan(metric_val) or math.isinf(metric_val)
            is_known_sentinel = metric_val in (999, 999.0, 999.9, 999.99, 9999, 99999,
                                               1000, 10000, -1, -999, -9999, 0)
            is_suspicious_extreme = (
                (lower and metric_val >= 500) or
                (lower and metric_val >= 50 and has_error_in_log) or
                (not lower and metric_val <= 0.001 and has_error_in_log)
            )
            is_sentinel = is_nan_or_inf or is_known_sentinel or is_suspicious_extreme
            if is_sentinel:
                log_tail = res.get("stdout", "")[-500:]
                log.result(f"Exp {n}: {model_name} → {metric_name}={metric_val:.6f} (SENTINEL — treating as failure)")
                res["success"] = False
                error = f"Sentinel metric value {metric_val} detected — script likely failed silently. Log tail: {log_tail[-300:]}"
                failure_reason = "bad_output_format"
                consecutive_crashes += 1
            else:
                secondary = " · ".join(f"{k}={v:.4f}" for k, v in all_metrics.items() if k != metric_name and k != "what_worked")
                sec_str = f" ({secondary})" if secondary else ""
                log.result(f"Exp {n}: {model_name} → {metric_name}={metric_val:.6f}{sec_str} ({res['elapsed']:.1f}s)")
                narrate(log_callback, "metrics_parsed", model=model_name, metrics=all_metrics, primary_metric=metric_name, val=metric_val)
                # Log train/test split if available
                tr_key = f"train_{metric_name}"
                te_key = f"test_{metric_name}"
                if tr_key in all_metrics and te_key in all_metrics:
                    log.engine(f"  ↳ {metric_name.upper()}: train={all_metrics[tr_key]:.4f} → test={all_metrics[te_key]:.4f}")
                    # Detect overfitting
                    _ov_ratio = abs(all_metrics[tr_key] - all_metrics[te_key]) / max(abs(all_metrics[te_key]), 1e-10)
                    if _ov_ratio > 0.3:
                        narrate(log_callback, "overfitting_warning", num=n, train=all_metrics[tr_key], test=all_metrics[te_key])
        
        if not res["success"]:
            if not error:
                error = res.get("error", "Unknown execution failure")
            if not failure_reason:
                failure_reason = classify_failure_reason(error)
                consecutive_crashes += 1

            # ── CRASH RECOVERY (Karpathy: trivial fix → rerun, fundamental → skip) ──
            if failure_reason in {"data_path", "invalid_hyperparameter", "bad_output_format", "missing_package", "runtime_error"}:
                narrate(log_callback, "crash_recovery", reason=failure_reason)
                repaired = auto_fix(train_py, error)
                repaired, fix_notes = apply_code_guardrails(repaired)
                if repaired.strip() and repaired.strip() != train_py.strip():
                    log.engine(f"Crash recovery [{failure_reason}] → auto-repair + retry")
                    # Auto-install any new packages the fix introduced
                    auto_install_packages(repaired, log)
                    retry_res = execute(repaired, csv_path, ws, n, data_sep=profile.get("detected_sep", ","))
                    if retry_res.get("success"):
                        res = retry_res
                        m = res["metrics"]
                        score = m
                        _primary_r = obj["metric"]
                        _search_r = [
                            _primary_r, _primary_r.lower(), _primary_r.upper(),
                            f"test_{_primary_r}", f"test_{_primary_r.lower()}",
                        ] + [k for k in ["rmse","r2","mape","mae","nse","auc","f1","accuracy"] if k != _primary_r.lower()]
                        for k in _search_r:
                            if k in m and _is_valid_metric(k, m[k]):
                                metric_val = float(m[k])
                                metric_name = _primary_r if k.lower().replace("test_","") == _primary_r.lower() else k
                                break
                        else:
                            for k, v in m.items():
                                if _is_valid_metric(k, v):
                                    metric_name = k
                                    metric_val = float(v)
                                    break
                        all_metrics = {k: float(v) for k, v in m.items() if _is_valid_metric(k, v)}
                        model_name = m.get("model") or _infer_model_name(repaired)
                        what_worked = m.get("what_worked", "") or "Recovered from auto-repair"
                        error = None
                        failure_reason = ""
                        consecutive_crashes = 0
                        train_py = repaired
                        log.result(f"Exp {n}: {model_name} → {metric_name}={metric_val:.6f} (recovered)")
                    else:
                        error = retry_res.get("error", error)
                        failure_reason = classify_failure_reason(error)

            if not res.get("success"):
                score = {"success": False, "error": error}
                log.err(f"Exp {n} crashed: {failure_reason or 'unknown'} — {error[:200]}")

                # If 3+ consecutive crashes, HARD RESET to last known good state
                if consecutive_crashes >= 3 and best_train_py and git_ok:
                    narrate(log_callback, "hard_reset")
                    log.engine("3 consecutive crashes — hard resetting to last known good train.py")
                    train_py = best_train_py
                    (ws / "train.py").write_text(f"DATA_PATH = {repr(str(csv_path))}\nDATA_SEP = {repr(profile.get('detected_sep', ','))}\nTIME_BUDGET = {TIME_BUDGET}\n\n{train_py}")
                    consecutive_crashes = 0

        if cancel_event and cancel_event.is_set():
            log.engine("Graceful stop — preserving current experiment results.")
            break

        # ── KEEP / DISCARD DECISION ──────────────────────────────
        revision = revise_after_iteration(program_md, train_py, score, error, history, domain_analysis=domain_analysis, obj=obj)
        keep = bool(revision["keep"] and res["success"])
        program_md = revision["new_program_md"]
        train_py_candidate = revision["new_train_py"]
        reasoning = revision["reasoning"]

        if res.get("success"):
            # Guard: treat NaN/inf metric as failure even if success=True
            if math.isnan(metric_val) or math.isinf(metric_val):
                res["success"] = False
                error = f"Metric value is {metric_val} — treating as failure"
                failure_reason = "bad_output_format"
                consecutive_crashes += 1
            is_first_success = best_val is None
            is_new_best = best_val is not None and (
                (lower and metric_val < best_val) or
                (not lower and metric_val > best_val)
            )
            is_worse = best_val is not None and (
                (lower and metric_val > best_val) or
                (not lower and metric_val < best_val)
            )

            # ENGINE OVERRIDE: force KEEP if this is the best score so far
            if not keep and (is_first_success or is_new_best):
                keep = True
                override_reason = "first successful result" if is_first_success else f"new best ({metric_val:.4f} vs {best_val:.4f})"
                reasoning = f"[ENGINE OVERRIDE: {override_reason}] {reasoning}"
                log.engine(f"Overriding DISCARD → KEEP: {override_reason}")

            # ENGINE VETO: force DISCARD if metric is worse than current best
            if keep and is_worse:
                keep = False
                reasoning = f"[ENGINE VETO: metric worse ({metric_val:.4f} vs best {best_val:.4f})] {reasoning}"
                log.engine(f"Overriding KEEP → DISCARD: metric regressed ({metric_val:.4f} vs {best_val:.4f})")

        status = "keep" if keep else ("crash" if not res["success"] else "discard")

        # Emit a structured result log that the frontend regex can parse
        if res.get("success"):
            _is_nb = (best_val is None) or (lower and metric_val < best_val) or (not lower and metric_val > best_val)
            _tag = "NEW BEST" if (keep and _is_nb) else ("KEEP" if keep else "DISCARD")
            log.result(f"Exp {n}: {model_name} → {metric_name}={metric_val:.6f} {_tag}")
        else:
            log.result(f"Exp {n}: CRASH → {failure_reason or 'error'}")

        history.append({
            "num": n,
            "status": status,
            "success": bool(res["success"]),
            "model": model_name,
            "metric_name": metric_name,
            "metric_val": float(metric_val),
            "all_metrics": all_metrics,
            "note": reasoning,
            "error": error or "",
            "failure_reason": failure_reason,
            "commit": "",
        })
        append_results_tsv(ws, n, model_name, metric_name, float(metric_val), status, reasoning or what_worked or "", all_metrics)
        prog = render_progress_png(ws, history, obj)
        if prog:
            artifacts["progress_png"] = prog

        # Clean up per-experiment plots from workspace (will regenerate final ones)
        for plot_name in ("predictions.png", "residuals.png", "feature_importance.png"):
            try:
                (ws / plot_name).unlink(missing_ok=True)
            except Exception:
                pass

        # program.md is ALWAYS updated (learnings accumulate regardless of keep/discard)
        (ws / "program.md").write_text(program_md)

        if keep:
            best_train_py = train_py
            better = best_val is None or (lower and metric_val < best_val) or ((not lower) and metric_val > best_val)
            if better:
                best_val = metric_val
                best = {
                    "num": n,
                    "model": model_name,
                    "metric_name": metric_name,
                    "metric_val": float(metric_val),
                    "all_metrics": all_metrics,
                    "all_history": history,
                }
                no_improve_rounds = 0
                # Preserve best model artifact — model.pkl gets overwritten each experiment
                _src_pkl = ws / "model.pkl"
                if _src_pkl.exists():
                    try:
                        shutil.copy2(_src_pkl, ws / "best_model.pkl")
                        log.engine(f"Saved best_model.pkl from exp {n}")
                    except Exception:
                        pass
            else:
                no_improve_rounds += 1
            insights.append(f"Exp {n}: KEEP — {reasoning}")
            train_py = train_py_candidate or train_py
            log.result(f"KEEP — {reasoning}")
            narrate(log_callback, "keep_decision", model=model_name, metric=metric_name, val=metric_val, reason=reasoning[:100])
            # Narrate improvement if there was a previous best
            if best_val is not None and best and best.get("metric_val") is not None:
                _prev_best = best["metric_val"]
                if _prev_best != metric_val and _prev_best != 0:
                    _pct = ((metric_val - _prev_best) / abs(_prev_best)) * 100
                    narrate(log_callback, "improvement", metric=metric_name, prev=_prev_best, curr=metric_val, pct=_pct)

            # GIT: commit this winning state (branch advances)
            if git_ok:
                sha = git_commit_experiment(ws, n, f"KEEP {model_name} {metric_name}={metric_val:.6f}")
                history[-1]["commit"] = sha
                log.engine(f"Git commit {sha}: KEEP")
        else:
            # Only count real experiments toward stagnation, not crashes/sentinels
            if res.get("success"):
                no_improve_rounds += 1
            insights.append(f"Exp {n}: DISCARD — {reasoning}")
            narrate(log_callback, "discard_decision", reason=reasoning[:120])

            # GIT: revert to last known good state (Karpathy: git reset --hard)
            if git_ok and best_train_py:
                discard_ok = git_discard_uncommitted(ws)
                if discard_ok:
                    log.engine(f"Git discard: reverted working tree to {git_get_commit_hash(ws)}")
                else:
                    log.engine("Git discard failed — restoring train.py from memory")
                    (ws / "train.py").write_text(
                        f"DATA_PATH = {repr(str(csv_path))}\nDATA_SEP = {repr(profile.get('detected_sep', ','))}\nTIME_BUDGET = {TIME_BUDGET}\n\n{best_train_py}"
                    )

            train_py = train_py_candidate or (best_train_py if best_train_py else train_py)
            log.engine(f"DISCARD — {reasoning}")

        # Detect stale code — if LLM returned identical train.py, force a rewrite
        prev_hash = hash(train_py.strip())
        train_py, post_notes = apply_code_guardrails(train_py)
        if post_notes:
            log.engine(f"Post-revision guardrails: {', '.join(sorted(set(post_notes)))}")

        if hash(train_py.strip()) == prev_hash and n < budget:
            # Check if the new code is the same as what we just ran
            current_on_disk = ""
            try:
                current_on_disk = (ws / "train.py").read_text().split("\n\n", 1)[-1].strip()
            except Exception:
                pass
            if train_py.strip() == current_on_disk:
                log.engine("STALE CODE DETECTED — LLM returned identical train.py. Forcing radical rewrite...")
                forced = ask(
                    "You are an ML engineer. The previous train.py produced IDENTICAL results twice. You MUST write a COMPLETELY DIFFERENT approach.",
                    f"""The current train.py keeps producing the same result: {metric_name}={metric_val:.4f}.

MANDATORY: Write a COMPLETELY different train.py using a DIFFERENT model/algorithm.
If the previous used Ridge/Linear, try: GradientBoosting, RandomForest, XGBoost, SVR, or ensemble.
If it used tree methods, try: neural net, elastic net, stacking, or a radically different preprocessing.

DO NOT copy any part of the old code. Start fresh.
Previous model: {model_name}
Available packages: {', '.join(AVAILABLE_PKGS) if AVAILABLE_PKGS else 'sklearn, numpy, pandas, joblib'}
Target: {obj.get('target', '')} | Metric: {metric_name} ({obj.get('direction', 'lower_is_better')})
Columns: {', '.join(profile['headers'])}

RULES: DATA_PATH and TIME_BUDGET are pre-injected variables. Use df = pd.read_csv(DATA_PATH).
Split into train/test. Report train_ and test_ prefixed metrics.
Final line: print(json.dumps(metrics)) with "model" key.
Generate plots (predictions.png, residuals.png) with dark theme.

Output ONLY a complete ```python block.""", 4200)
                train_py = extract_code(forced)
                train_py, _ = apply_code_guardrails(train_py)
                log.engine(f"Forced rewrite complete — new approach ready")

        (ws / "train.py").write_text(f"DATA_PATH = {repr(str(csv_path))}\nDATA_SEP = {repr(profile.get('detected_sep', ','))}\nTIME_BUDGET = {TIME_BUDGET}\n\n{train_py}")
        write_workspace_requirements(ws, train_py)

        # ── STOP CONDITIONS ──────────────────────────────────────
        # Good-enough threshold: stop in any mode
        thresh = GOOD_ENOUGH.get(metric_name) or GOOD_ENOUGH.get(metric_name.lower())
        ge_str = obj.get("good_enough", "")
        # Only apply LLM's good_enough if it mentions the same metric we're tracking
        # AND is stricter than our built-in. This prevents "RMSE < 25000" from stopping
        # a MAPE-optimized run after 1 experiment.
        if ge_str and metric_name.lower() in ge_str.lower():
            ge_nums = re.findall(r'[\d,]+\.?\d*', ge_str.replace(',', ''))
            if ge_nums:
                try:
                    parsed_t = float(ge_nums[0])
                    if parsed_t > 0:
                        # Only replace if stricter (lower for lower_is_better, higher for higher_is_better)
                        if thresh is None:
                            thresh = parsed_t
                        elif lower and parsed_t < thresh:
                            thresh = parsed_t
                        elif not lower and parsed_t > thresh:
                            thresh = parsed_t
                except ValueError:
                    pass
        if keep and thresh and ((lower and metric_val <= thresh) or ((not lower) and metric_val >= thresh)):
            narrate(log_callback, "good_enough", metric=metric_name, val=metric_val)
            log.engine(f"Hit good-enough threshold ({thresh}) on {metric_name}. Mission accomplished.")
            break

        # In continuous mode: stagnation triggers STRATEGY PIVOT, not exit
        if continuous and no_improve_rounds >= 5:
            narrate(log_callback, "stagnation")
            log.engine(f"Stagnation ({no_improve_rounds} rounds) — pivoting strategy, NOT stopping")
            pivot_insight = f"STAGNATION ALERT: {no_improve_rounds} rounds without improvement. RADICALLY change approach."
            insights.append(pivot_insight)
            program_md = write_program_md(profile, obj, history, insights)
            (ws / "program.md").write_text(program_md)
            no_improve_rounds = 0  # reset after pivot
            log.engine("Rewrote program.md with pivot directive")
        elif not continuous and stagnation_limit and no_improve_rounds >= stagnation_limit:
            log.engine(f"Stopping: stagnated for {no_improve_rounds} rounds (limit={stagnation_limit}).")
            break

    # ── REPORT + DEPLOY ──────────────────────────────────────────
    total_experiments = len(history)
    kept = sum(1 for h in history if h["status"] == "keep")
    crashed = sum(1 for h in history if h["status"] == "crash")
    log.engine(f"\nCompleted {total_experiments} experiments: {kept} kept, {crashed} crashed, {total_experiments - kept - crashed} discarded")

    # Fallback: if no KEEP but we have successful experiments, pick the best one
    if not best:
        successful = [h for h in history if h.get("success") and h.get("metric_val") is not None]
        if successful:
            if lower:
                best_h = min(successful, key=lambda h: h["metric_val"])
            else:
                best_h = max(successful, key=lambda h: h["metric_val"])
            best = {
                "num": best_h["num"],
                "model": best_h["model"],
                "metric_name": best_h["metric_name"],
                "metric_val": float(best_h["metric_val"]),
                "all_metrics": best_h.get("all_metrics", {}),
                "all_history": history,
            }
            log.engine(f"Fallback best: {best['model']} {best['metric_name']}={best['metric_val']:.6f} (exp {best['num']})")

    if best:
        narrate(log_callback, "final_report")
        best["all_history"] = history
        report = generate_report(ws, obj, profile, history, best)
        log.engine("Packaging deployment...")
        deploy = package_deployment(ws, best, obj, profile)
        diagnostics = build_run_diagnostics(obj, profile, history, best, deploy)

        # Final git tag
        if git_ok:
            try:
                _git(ws, "tag", f"best-{best['metric_name']}-{best['metric_val']:.4f}")
            except Exception:
                pass

        log.sys(f"\n{'═'*50}")
        am = best.get("all_metrics", {})
        sec = " · ".join(f"{k}={v:.4f}" for k, v in am.items() if k != best["metric_name"])
        sec_str = f" ({sec})" if sec else ""
        log.sys(f"DONE — Best: {best['model']} {best['metric_name']}={best['metric_val']:.6f}{sec_str} (exp {best['num']})")
        log.sys(f"Total experiments: {total_experiments} | Kept: {kept} | Crashed: {crashed}")
        if git_ok:
            log.sys(f"Git HEAD: {git_get_commit_hash(ws)}")
        log.sys(f"Workspace: {ws}")
        log.sys(f"{'═'*50}")
    else:
        report = "All experiments failed."
        deploy = None
        diagnostics = build_run_diagnostics(obj, profile, history, best, deploy)
        log.err("All experiments failed.")

    artifacts["program_md"] = str(ws / "program.md")
    artifacts["prepare_py"] = str(ws / "prepare.py")
    artifacts["analysis_ipynb"] = write_analysis_notebook(ws, profile, obj)
    artifacts["results_tsv"] = str(ws / "results.tsv")
    artifacts["train_py"] = str(ws / "train.py")
    if (ws / "final_report.md").exists():
        artifacts["final_report_md"] = str(ws / "final_report.md")

    # Generate comprehensive final plots
    log.engine("Generating final visualizations...")
    final_plots = render_final_plots(ws, history, obj, best)
    artifacts.update(final_plots)
    if final_plots:
        log.engine(f"Generated {len(final_plots)} plots: {', '.join(final_plots.keys())}")

    # ── Predictions summary narration ─────────────────────────────
    pred_csv = ws / "predictions.csv"
    if pred_csv.exists() and best:
        try:
            import pandas as _pd
            _preds = _pd.read_csv(pred_csv, nrows=500)
            _ncols = [c for c in _preds.columns if c.lower() in ("actual","predicted","error","pct_error")]
            _nrows = len(_preds)
            _summary = f"Forecast table ready — {_nrows} rows"
            if "actual" in _preds.columns and "predicted" in _preds.columns:
                _errs = (_preds["predicted"] - _preds["actual"])
                _mae = _errs.abs().mean()
                _summary += f" · MAE={_mae:.4f}"
            narrate(log_callback, "predictions_ready", summary=_summary, n_rows=_nrows)
        except Exception:
            pass
    artifacts["predictions_csv"] = str(pred_csv) if pred_csv.exists() else ""

    return dict(workspace=str(ws), objective=obj, profile=profile,
        history=history, best=best, deploy_path=deploy, report=report,
        diagnostics=diagnostics, executive_brief=diagnostics.get("executive_brief", ""),
        artifacts=artifacts,
        train_script=str(ws / "train.py"), results_tsv=str(ws / "results.tsv"),
        continuous_mode=continuous, total_experiments=total_experiments,
        token_usage=get_token_usage())

# ── CHAT WITH DATA ─────────────────────────────────────────────
def chat_with_data(message: str, context: dict, api_key: str, provider: str = "claude", model: str = None) -> str:
    """Free-form conversational AI about the current dataset/model."""
    # ── Build rich context block ──────────────────────────────────
    ctx_lines = []
    if context.get("filename"):
        ctx_lines.append(f"**Dataset:** {context['filename']}")

    profile = context.get("profile") or {}
    if profile.get("n_rows"):
        ctx_lines.append(f"**Shape:** {profile['n_rows']:,} rows × {profile.get('n_cols', '?')} columns")
    if profile.get("headers"):
        cols = profile["headers"][:30]
        ctx_lines.append(f"**Columns:** {', '.join(cols)}" + (f" (+{len(profile['headers'])-30} more)" if len(profile['headers']) > 30 else ""))

    # Column type info
    if profile.get("types"):
        num_cols = [c for c,t in profile["types"].items() if t == "numeric"]
        cat_cols = [c for c,t in profile["types"].items() if t != "numeric"]
        if num_cols:
            ctx_lines.append(f"**Numeric cols:** {', '.join(num_cols[:15])}")
        if cat_cols:
            ctx_lines.append(f"**Categorical cols:** {', '.join(cat_cols[:15])}")

    # Missing values
    if profile.get("nulls"):
        high_null = [(c, v) for c, v in profile["nulls"].items() if isinstance(v, (int, float)) and v > 0]
        if high_null:
            null_str = ", ".join(f"{c}: {v}" for c, v in sorted(high_null, key=lambda x: -x[1])[:8])
            ctx_lines.append(f"**Missing values:** {null_str}")

    obj = context.get("objective") or {}
    if obj.get("task"):
        ctx_lines.append(f"**ML Task:** {obj.get('task')} | **Target:** {obj.get('target','?')} | **Metric:** {obj.get('metric','?')}")

    if context.get("best"):
        b = context["best"]
        bam = b.get("all_metrics") or {}
        pk = b.get("metric_name", "metric")
        pv = b.get("metric_val")
        metric_str = f"{pk}={pv:.4f}" if isinstance(pv, (int, float)) else f"{pk}={pv}"
        extra = [f"{k}={v:.4f}" for k, v in bam.items() if k != pk and k not in (f"train_{pk}", f"test_{pk}") and isinstance(v, (int, float))][:4]
        extra_str = " | " + " | ".join(extra) if extra else ""
        ctx_lines.append(f"**Best model:** {b.get('model','?')} — {metric_str}{extra_str}")

    if context.get("history"):
        h = context["history"]
        kept = [e for e in h if e.get("success") and e.get("status") != "discard"]
        models_tried = list({e.get("model","?") for e in h if e.get("model")})[:8]
        ctx_lines.append(f"**Experiments:** {len(h)} run, {len(kept)} kept — models tried: {', '.join(models_tried)}")

    if context.get("report"):
        report_snip = str(context["report"])[:600]
        ctx_lines.append(f"**Final report summary:** {report_snip}")

    # ── System prompt ──────────────────────────────────────────────
    system = """You are 19, an expert AI data scientist built into 19Labs — an autonomous ML research platform.
You are sharp, direct, and genuinely helpful. You know statistics, machine learning, feature engineering, model selection, and data analysis deeply.

BEHAVIOR:
- Answer specifically using the session context below — reference actual column names, metric values, and model names when relevant
- If the user asks about their data, explain patterns, correlations, or potential issues concretely
- If asked about model results, explain what the metrics mean in plain terms and whether the performance is good for the task
- If asked what to do next, give a concrete, actionable recommendation
- Never be vague. Never say "it depends" without following up with specifics
- Keep responses concise but complete. Use markdown for structure when helpful
- If no dataset is loaded yet, guide the user to upload one and explain what 19Labs can do"""

    if ctx_lines:
        system += "\n\n## Current Session\n" + "\n".join(ctx_lines)
    else:
        system += "\n\n## Current Session\nNo dataset loaded yet."

    # ── Build message history for multi-turn conversation ─────────
    history = context.get("chat_history") or []
    messages = []
    for turn in history[-10:]:  # last 10 turns for context window efficiency
        role = turn.get("role")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    try:
        if provider == "openai" and OpenAI:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model or "gpt-4o",
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=1000
            )
            return resp.choices[0].message.content
        else:
            client = Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=model or CLAUDE_MODEL, max_tokens=1000,
                system=system,
                messages=messages
            )
            return resp.content[0].text
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid" in err.lower() and "key" in err.lower():
            return "❌ Invalid API key — please update your key in Settings (top-right ⚙)."
        if "429" in err or "rate" in err.lower():
            return "⏱ Rate limit hit — wait a moment and try again."
        if "insufficient_quota" in err or "quota" in err.lower():
            return "💳 API quota exceeded — check your billing at your provider dashboard."
        return f"I couldn't process that — {err[:120]}"


# ── INFERENCE SERVER GENERATOR ─────────────────────────────────
def generate_inference_server(train_py: str, best: dict, obj: dict, api_key: str, provider: str = "claude") -> dict:
    """Generate a complete deployable FastAPI inference server from the trained model code."""
    # Use the module-level client so we don't need to re-authenticate
    _init_client(api_key, provider)

    model_name = best.get("model", "MLModel")
    task = obj.get("task", "regression")
    target = obj.get("target", "target")
    metric = obj.get("metric", "accuracy")

    system_prompt = "You are a senior ML engineer. Generate clean, production-ready FastAPI server code."
    user_prompt = f"""Generate a production-ready FastAPI inference server for this trained ML model.

TRAINING CODE (first 3500 chars):
```python
{train_py[:3500]}
```

MODEL: {model_name} | TASK: {task} | TARGET: {target} | METRIC: {metric}

Generate a complete inference_server.py that:
1. Loads the trained model at startup — scan for .pkl, .joblib, .pt, .h5, best_model.* in the same directory
2. POST /predict — accepts JSON dict of features, returns {{"prediction": value, "probability": null_or_float}}
3. GET /health — returns {{"status": "ok", "model": "{model_name}", "task": "{task}"}}
4. Handles missing/extra input fields gracefully
5. Applies the EXACT same feature preprocessing from the training code
6. Works with uvicorn on Railway ($PORT env var)

Also generate minimal requirements_txt (inference only, not training) and a Dockerfile using python:3.11-slim.

Return ONLY valid JSON with keys: inference_server_py, requirements_txt, dockerfile
No markdown fences, no explanation — just the JSON object."""

    try:
        raw = ask(system_prompt, user_prompt, 3000)
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r'\{[\s\S]*\}', raw)
            return json.loads(m.group()) if m else {"error": "Could not parse response"}
    except Exception as e:
        return {"error": str(e)}


# ── CLI ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="19Labs Autoresearch Engine — Karpathy-discipline ML research loop")
    ap.add_argument("csv")
    ap.add_argument("--hint",      default="")
    ap.add_argument("--budget",    type=int, default=6)
    ap.add_argument("--reliability-mode", default="balanced")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--api-key",   default=None)
    ap.add_argument("--continuous", action="store_true", help="NEVER STOP mode — runs until killed or good-enough")
    a = ap.parse_args()
    r = run_research(a.csv, workspace=a.workspace or f"./ws_{int(time.time())}",
        budget=a.budget, user_hint=a.hint, api_key=a.api_key,
        reliability_mode=a.reliability_mode, continuous=a.continuous)
    print(f"\nBest: {r['best']['model'] if r['best'] else 'None'}")
    print(f"Total experiments: {r['total_experiments']}")
    print(f"Workspace: {r['workspace']}")
    print(f"Deploy: {r['deploy_path']}")
