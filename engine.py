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

# Packages too large/slow to install mid-experiment (>200MB or >3min install).
# Map them to lighter drop-in alternatives where possible.
_HEAVY_PACKAGE_REDIRECTS = {
    "tensorflow": None,   # skip — use torch or sklearn instead
    "keras": None,        # skip — use torch or sklearn instead
    "tf": None,
    "torch": None,        # skip — too large for auto-install; use sklearn/xgboost
    "torchvision": None,
    "torchaudio": None,
    "jax": None,
    "jaxlib": None,
    "transformers": None, # skip — too large
    "sentence_transformers": None,
    "sentence-transformers": None,
    "datasets": None,
    "diffusers": None,
    "cv2": None,          # opencv — large
    "opencv-python": None,
    "opencv-python-headless": None,
}

_INSTALL_TIMEOUT = 60  # seconds — if pip takes longer, skip the package

def auto_install_packages(code: str, log=None) -> list:
    """Parse imports from generated code and pip-install any missing packages."""
    import importlib
    modules = detect_imported_modules(code)
    to_install = []
    for mod in modules:
        pip_name = MODULE_TO_PIP.get(mod, mod)
        if pip_name in _installed_session:
            continue
        # Skip standard library and internal modules
        if mod in {"json", "os", "sys", "re", "time", "math", "random", "pathlib",
                   "collections", "itertools", "functools", "typing", "abc", "io",
                   "csv", "datetime", "warnings", "copy", "string", "struct",
                   "hashlib", "base64", "urllib", "http", "threading", "subprocess"}:
            continue
        # Redirect or skip heavy packages
        if pip_name in _HEAVY_PACKAGE_REDIRECTS:
            redirect = _HEAVY_PACKAGE_REDIRECTS[pip_name]
            if redirect is None:
                if log:
                    log.engine(f"Skipping {pip_name} (too large for auto-install — engine will use lighter alternative)")
                continue
            pip_name = redirect
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
                log.engine(f"Install timed out ({_INSTALL_TIMEOUT}s) for {pkg} — skipping")
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
CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o"
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
        gitignore.write_text("__pycache__/\n*.pyc\nmodel.pkl\nrun.log\n*.zip\n")
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
            col_info.update(
                mean=round(float(num.mean()), 4),
                std=round(float(num.std()), 4),
                min=round(float(num.min()), 4),
                max=round(float(num.max()), 4),
                p25=round(float(num.quantile(0.25)), 4),
                p50=round(float(num.quantile(0.50)), 4),
                p75=round(float(num.quantile(0.75)), 4),
                skew=round(float(num.skew()), 3),
                # Serial correlation — high value suggests time-series ordering
                serial_corr=round(float(num.autocorr(lag=1)) if len(num) > 2 else 0.0, 3),
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
                top_correlations.append({"cols": [c1, c2], "corr": round(float(v), 3)})
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

    return dict(
        path=str(csv_path), rows=len(df), cols=len(df.columns),
        headers=list(df.columns), columns=cols,
        numeric=numeric_cols, categorical=cat_cols,
        datetime=datetime_cols, text=text_cols, filepath=filepath_cols,
        signals=signals,
        class_balance=class_balance,
        top_correlations=top_correlations,
        detected_sep=detected_sep,
        target_candidates=[c["name"] for c in cols if c["type"] == "numeric" and c["unique"] > 10][:5],
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
def _init_client(api_key, provider="claude"):
    global _client, _provider
    _provider = (provider or "claude").lower()
    if _provider == "openai":
        if OpenAI is None:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        _client = OpenAI(api_key=api_key)
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
    if "connection error" in msg or "connection refused" in msg:
        return "Cannot reach the API. Check your internet connection, VPN, or firewall."
    if "authentication" in msg or "401" in msg:
        return "Authentication failed. Your API key may be expired or revoked."
    return None

def ask(system, user, max_tokens=3000):
    last_err = None
    for attempt in range(1, 4):
        try:
            if _provider == "openai":
                r = _client.chat.completions.create(
                    model=OPENAI_MODEL,
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
                r = _client.messages.create(
                    model=CLAUDE_MODEL,
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

# ── INFER OBJECTIVE ────────────────────────────────────────────
def infer_objective(profile, hint="", domain_analysis=""):
    resp = ask(
        "You are 19Labs. Given a deep expert domain analysis, produce precise ML objective parameters.",
        f"""Based on the expert domain analysis below, extract the exact ML objective parameters.

EXPERT DOMAIN ANALYSIS:
{domain_analysis or "(not available — reason from profile)"}

DATASET SUMMARY:
- Rows: {profile['rows']:,} | Cols: {profile['cols']}
- Columns: {', '.join(profile['headers'])}
- Signals: {'; '.join(profile.get('signals', []))}
- Target candidates: {', '.join(profile['target_candidates'])}
{"- USER HINT: " + hint if hint else ""}

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
    return dict(domain=g("DOMAIN") or "General",
        task   =g("TASK")   or "Regression",
        target =g("TARGET") or (tc[0] if tc else profile["headers"][-1]),
        metric =g("METRIC") or "rmse",
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

def discover_user_need(csv_path, user_hint="", api_key=None, provider="claude"):
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("No API key.")
    _init_client(key, provider)

    profile = profile_dataset(csv_path)
    obj = infer_objective(profile, user_hint)

    advice_raw = ask(
        "You are a product-minded ML research copilot. Help clarify user intent before training.",
        f"""Given this dataset profile and initial objective inference, propose the best next direction.

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
  "first_iteration_plan": "one concise paragraph"
}}

DATA PROFILE:
- rows: {profile['rows']}
- cols: {profile['cols']}
- headers: {profile['headers']}
- numeric: {profile['numeric']}
- categorical: {profile['categorical']}
- datetime: {profile['datetime']}
- text_columns: {profile.get('text', [])}
- signals: {profile.get('signals', [])}

INFERRED OBJECTIVE:
- task: {obj['task']}
- target: {obj['target']}
- metric: {obj['metric']} ({obj['direction']})
- domain: {obj['domain']}
- confidence: {obj['confidence']}
- reasoning: {obj['reasoning']}

USER HINT:
{user_hint or "(none)"}
""",
        1200
    )
    advice = _extract_json_blob(advice_raw)
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

Fix the error. Keep the model/approach if possible. Output ONLY fixed ```python code.""", 3000))
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
- You have access to the full Python ML ecosystem. Any package you import will be auto-installed.
- Pre-installed: sklearn, xgboost, lightgbm, catboost, pandas, numpy, matplotlib, scipy, statsmodels, optuna, shap, joblib
- Available on demand: transformers, torch, tensorflow, prophet, sentence-transformers,
  hdbscan, umap-learn, opencv-python-headless, Pillow, and more.
- Use whatever is genuinely best for the task. Do NOT artificially limit yourself.

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
    code = ask(
        "You write complete production-grade Python training scripts. "
        "You are a senior ML engineer — you know exactly what approach fits each dataset type and domain. "
        "You never write generic code when expert-level code is possible.",
        f"""Write `train.py` — experiment {exp_num} — for this specific domain and task.

ENVIRONMENT:
- Full Python ML ecosystem. Any import is auto-installed before execution.
- Pre-installed: sklearn, xgboost, lightgbm, catboost, pandas, numpy, matplotlib, scipy, statsmodels, optuna, shap, joblib
- On-demand (auto-install): transformers, torch, tensorflow, prophet, sentence-transformers, hdbscan, umap-learn, etc.
- CRITICAL: Use what genuinely fits. For NLP → transformers/sentence-transformers. For time series → prophet/statsmodels.
  For imbalanced → SMOTE+class weights. For tabular → gradient boosting + optuna. For images → torch/keras.

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

EXECUTION POLICY:
- Reliability mode: {obj.get('reliability_mode', 'balanced')}
- {obj.get('execution_policy', 'Balance reliability and performance.')}

KARPATHY DISCIPLINE (MANDATORY):
- DATA_PATH, DATA_SEP, and TIME_BUDGET are Python variables pre-defined BEFORE your code runs (injected at line 1).
  DO NOT redefine them. DO NOT use os.environ.get(). Use them directly:
  `df = pd.read_csv(DATA_PATH, sep=DATA_SEP)` — this is the ONLY way to load data.
  DATA_SEP is already set to the correct delimiter (e.g. ',' or ';' or '\\t').
- TIME_BUDGET is also pre-defined. DO NOT redefine it. Your entire training MUST complete within it.
  Add a wall-clock check: `import time; _start = time.time()` at top, and periodically
  check `if time.time() - _start > TIME_BUDGET * 0.9: break` in any training loops.
  This is NON-NEGOTIABLE. Timeout = automatic DISCARD.
- Robust preprocessing (nulls, categoricals, datetime).
- Deterministic behavior (set random seeds).
- Save model via `joblib.dump(model, 'model.pkl')`.
- ALL output goes to stdout. Final line MUST be `print(json.dumps(metrics))` where metrics is a dict with these keys:
  REQUIRED:
  - "model": string naming the algorithm (e.g. "Ridge", "XGBoost", "LightGBM")
  - "{obj.get('metric', 'rmse')}": float — the PRIMARY metric on TEST set
  ALWAYS INCLUDE (compute all on TEST set):
  - "train_rmse": float, "test_rmse": float — train vs test RMSE (ALWAYS include both)
  - "train_mape": float, "test_mape": float — train vs test MAPE
  - "train_r2": float, "test_r2": float — train vs test R²
  - "rmse": float, "mape": float, "mae": float, "r2": float, "nse": float — test set values
  - "nse": float — Nash-Sutcliffe efficiency on test set
  - "what_worked": string
  IMPORTANT: Always split data into train/test, compute metrics on BOTH, and report both.
  The primary metric value MUST be computed on the TEST set, never the training set.
  Example: print(json.dumps({{"model": "XGBoost", "rmse": 4500.0, "train_rmse": 3200.0, "test_rmse": 4500.0, "train_r2": 0.97, "test_r2": 0.91, "r2": 0.91, "mape": 8.2, "nse": 0.91, "what_worked": "feature engineering"}}))
  NEVER use try/except to print fallback metrics. Let exceptions crash — the system handles recovery.
- GENERATE PLOTS: After training, save these plots using matplotlib (use Agg backend):
  1. `predictions.png` — scatter plot of y_true vs y_pred on the test set, with a diagonal reference line
  2. `residuals.png` — residual distribution (histogram) or residuals vs predicted
  3. `feature_importance.png` — if the model supports it (tree models, etc.)
  Use dark style: fig.set_facecolor('#0d1117'), ax.set_facecolor('#161b22'), white text/labels.
  Save all plots to the current working directory. If plot generation fails, let it pass silently (plots are optional, metrics are not).
- This is the ONLY file you edit. Everything must be self-contained in this script.

Output ONLY a complete ```python block.""",
        4200
    )
    clean, _ = apply_code_guardrails(extract_code(code))
    return clean

def revise_after_iteration(program_md, train_py, score, error, history, domain_analysis=""):
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
- Final line: print(json.dumps(metrics)) — MUST include "model" key PLUS train_ and test_ prefixed metrics.
  Required keys: "train_rmse", "test_rmse", "train_r2", "test_r2", "rmse" (=test), "r2" (=test), and any other applicable: mape, mae, nse.
- GENERATE PLOTS (matplotlib, Agg backend, dark style with facecolor '#0d1117'):
  1. predictions.png — y_true vs y_pred scatter on test set, with diagonal line
  2. residuals.png — residual histogram or residuals vs predicted
  3. feature_importance.png — if model supports it (tree models etc.)
  Wrap plot generation in try/except so it never blocks metrics output.
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
        7000
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
        # Log that LLM didn't produce structured output — caller will detect stale code
        import sys as _sys
        print(f"[engine] revise_after_iteration: LLM fallback for {used_fallback} — response may be malformed", file=_sys.stderr)
    tm, _ = apply_code_guardrails(tm)

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

def render_progress_png(ws, history, obj):
    """Single progress chart — updated every experiment."""
    ws = pathlib.Path(ws)
    p = ws / "progress.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    rows = [h for h in history if h.get("success")]
    if not rows:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No successful experiments yet" if history else "No experiments yet",
                ha="center", va="center", fontsize=14, color="#888")
        ax.set_facecolor("#0d1117"); fig.set_facecolor("#0d1117")
        ax.axis("off"); plt.tight_layout(); plt.savefig(p, dpi=140, facecolor="#0d1117"); plt.close(fig)
        return str(p)

    _apply_dark_style(plt)
    metric = obj.get("metric", "metric")
    lower = obj.get("direction", "lower_is_better") == "lower_is_better"
    xs = [int(h["num"]) for h in rows]
    ys = [float(h["metric_val"]) for h in rows]
    running = []
    cur = None
    for y in ys:
        cur = y if cur is None else (min(cur, y) if lower else max(cur, y))
        running.append(cur)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, ys, "o-", alpha=0.4, color="#8b949e", label="Each experiment")
    ax.plot(xs, running, "o-", linewidth=2.5, color="#58a6ff", label="Running best")
    ax.fill_between(xs, ys, running, alpha=0.06, color="#58a6ff")
    ax.set_title(f"Experiment Progress — {metric.upper()}", fontsize=14, fontweight="bold", color="white")
    ax.set_xlabel("Experiment #"); ax.set_ylabel(metric.upper())
    ax.legend(framealpha=0.3)
    plt.tight_layout(); plt.savefig(p, dpi=160, facecolor="#0d1117"); plt.close(fig)
    return str(p)


def _apply_dark_style(plt):
    plt.rcParams.update({
        "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e", "ytick.color": "#8b949e",
        "text.color": "#c9d1d9", "grid.color": "#21262d", "grid.alpha": 0.5,
        "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
    })


def render_final_plots(ws, history, obj, best):
    """Generate clean, professional final plots after all experiments."""
    ws = pathlib.Path(ws)
    generated = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return generated

    _apply_dark_style(plt)
    rows = [h for h in history if h.get("success")]
    if not rows:
        return generated

    metric = obj.get("metric", "metric")
    lower = obj.get("direction", "lower_is_better") == "lower_is_better"
    BLUE, PURPLE, AMBER, RED, GRAY = "#58a6ff", "#a78bfa", "#d29922", "#f85149", "#8b949e"
    palette = [BLUE, PURPLE, AMBER, "#79c0ff", "#bc8cff", GRAY, "#e3b341", "#f0883e"]

    # ── 1. TRAIN vs TEST — separate chart per metric (same scale) ─
    try:
        metric_pairs = [
            ("rmse", "train_rmse", "test_rmse", "RMSE"),
            ("r2",   "train_r2",   "test_r2",   "R²"),
            ("mape", "train_mape", "test_mape", "MAPE"),
        ]
        available_pairs = []
        for _, tk, vk, label in metric_pairs:
            if any(h.get("all_metrics", {}).get(tk) is not None for h in rows):
                available_pairs.append((tk, vk, label))

        if available_pairs:
            n_panels = len(available_pairs)
            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
            if n_panels == 1:
                axes = [axes]

            xs = [h["num"] for h in rows]
            x_pos = np.arange(len(xs))
            w = 0.35

            for i, (tk, vk, label) in enumerate(available_pairs):
                ax = axes[i]
                trains = [h.get("all_metrics", {}).get(tk, 0) for h in rows]
                tests = [h.get("all_metrics", {}).get(vk, 0) for h in rows]
                ax.bar(x_pos - w/2, trains, w, label="Train", color=BLUE, alpha=0.85)
                ax.bar(x_pos + w/2, tests, w, label="Test", color=PURPLE, alpha=0.85)
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"#{x}" for x in xs], fontsize=9)
                ax.set_title(f"Train vs Test — {label}", fontsize=13, fontweight="bold", color="white")
                ax.set_ylabel(label)
                ax.legend(framealpha=0.3, fontsize=9)
                for j, (tr, te) in enumerate(zip(trains, tests)):
                    ax.text(x_pos[j] - w/2, tr, f"{tr:.1f}" if tr > 10 else f"{tr:.3f}",
                            ha="center", va="bottom", fontsize=7, color="#8b949e")
                    ax.text(x_pos[j] + w/2, te, f"{te:.1f}" if te > 10 else f"{te:.3f}",
                            ha="center", va="bottom", fontsize=7, color="#8b949e")

            plt.tight_layout()
            p = ws / "train_test.png"
            plt.savefig(p, dpi=160, facecolor="#0d1117"); plt.close(fig)
            generated["train_test_png"] = str(p)
    except Exception:
        pass

    # ── 2. MODEL COMPARISON — primary metric + R² ─────────────────
    try:
        from collections import OrderedDict
        model_best = OrderedDict()
        for h in rows:
            m = h.get("model", "Unknown")
            v = float(h["metric_val"])
            am = h.get("all_metrics", {})
            if m not in model_best or (lower and v < model_best[m]["val"]) or (not lower and v > model_best[m]["val"]):
                model_best[m] = {"val": v, "am": am}

        if len(model_best) >= 2:
            sorted_models = sorted(model_best.items(), key=lambda x: x[1]["val"], reverse=not lower)
            names = [m[:25] for m, _ in sorted_models]
            vals = [d["val"] for _, d in sorted_models]
            bar_c = [palette[i % len(palette)] for i in range(len(names))]
            bar_c[0] = BLUE

            r2_vals = [d["am"].get("test_r2") or d["am"].get("r2") for _, d in sorted_models]
            has_r2 = all(v is not None for v in r2_vals)
            n_panels = 2 if has_r2 else 1
            fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, max(3, len(names) * 0.7 + 1.5)))
            if n_panels == 1:
                axes = [axes]

            ax = axes[0]
            y_pos = np.arange(len(names))
            bars = ax.barh(y_pos, vals, color=bar_c, alpha=0.85, height=0.6)
            ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=10)
            ax.set_xlabel(metric.upper())
            ax.set_title(f"Model Comparison — {metric.upper()}", fontsize=13, fontweight="bold", color="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_width() + max(vals) * 0.01, bar.get_y() + bar.get_height()/2,
                        f"{v:,.1f}", va="center", fontsize=9, color="#c9d1d9")

            if has_r2:
                ax2 = axes[1]
                bars2 = ax2.barh(y_pos, r2_vals, color=bar_c, alpha=0.85, height=0.6)
                ax2.set_yticks(y_pos); ax2.set_yticklabels(names, fontsize=10)
                ax2.set_xlabel("R²"); ax2.set_xlim(0, 1.05)
                ax2.set_title("Model Comparison — R²", fontsize=13, fontweight="bold", color="white")
                ax2.axvline(0.9, color=BLUE, linewidth=0.8, linestyle="--", alpha=0.5, label="R²=0.9")
                for bar, v in zip(bars2, r2_vals):
                    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                             f"{v:.3f}", va="center", fontsize=9, color="#c9d1d9")
                ax2.legend(framealpha=0.3, fontsize=8)

            plt.tight_layout()
            p = ws / "model_comparison.png"
            plt.savefig(p, dpi=160, facecolor="#0d1117"); plt.close(fig)
            generated["model_comparison_png"] = str(p)
    except Exception:
        pass

    # ── 3. BEST MODEL SCORECARD — train vs test per metric ────────
    try:
        best_am = (best or {}).get("all_metrics", {})
        score_metrics = []
        for base in ["rmse", "mape", "r2", "mae", "nse"]:
            tr = best_am.get(f"train_{base}")
            te = best_am.get(f"test_{base}") or best_am.get(base)
            if tr is not None and te is not None:
                score_metrics.append((base.upper(), tr, te))

        if score_metrics:
            n = len(score_metrics)
            fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
            if n == 1:
                axes = [axes]

            model_name = (best or {}).get("model", "Best Model")
            fig.suptitle(f"{model_name} — Train vs Test", fontsize=14, fontweight="bold", color="white", y=0.98)

            for i, (name, tr, te) in enumerate(score_metrics):
                ax = axes[i]
                bars = ax.bar([0, 1], [tr, te], color=[BLUE, PURPLE], alpha=0.85, width=0.6)
                ax.set_xticks([0, 1]); ax.set_xticklabels(["Train", "Test"], fontsize=10)
                ax.set_title(name, fontsize=12, fontweight="bold", color="white")
                for bar, v in zip(bars, [tr, te]):
                    fmt = f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}"
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            fmt, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c9d1d9")
                ax.set_xlim(-0.5, 1.5)

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            p = ws / "metrics_overview.png"
            plt.savefig(p, dpi=160, facecolor="#0d1117"); plt.close(fig)
            generated["metrics_overview_png"] = str(p)
    except Exception:
        pass

    # ── 4. EXPERIMENT TIMELINE ────────────────────────────────────
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        xs = [h["num"] for h in rows]
        test_vals = [float(h["metric_val"]) for h in rows]
        train_key = f"train_{metric}"
        train_vals = [h.get("all_metrics", {}).get(train_key) for h in rows]
        has_train = any(v is not None for v in train_vals)

        running = []
        cur = None
        for y in test_vals:
            cur = y if cur is None else (min(cur, y) if lower else max(cur, y))
            running.append(cur)

        if has_train:
            tv = [v if v is not None else 0 for v in train_vals]
            ax.plot(xs, tv, "s--", color=BLUE, alpha=0.6, label="Train", markersize=6)
        ax.plot(xs, test_vals, "o-", color=PURPLE, alpha=0.7, label="Test", markersize=6)
        ax.plot(xs, running, "o-", linewidth=2.5, color=BLUE, label="Running Best", markersize=7)
        ax.fill_between(xs, test_vals, running, alpha=0.06, color=BLUE)

        crashes = [h for h in history if not h.get("success")]
        if crashes:
            cx = [h["num"] for h in crashes]
            cy = [ax.get_ylim()[0]] * len(cx)
            ax.scatter(cx, cy, marker="x", color=RED, s=60, zorder=5, label="Crashed")

        ax.set_title(f"Experiment Timeline — {metric.upper()}", fontsize=14, fontweight="bold", color="white")
        ax.set_ylabel(metric.upper()); ax.legend(framealpha=0.3, fontsize=9)

        ax2 = axes[1]
        all_models = list(set(h.get("model", "?") for h in history))
        model_color = {m: palette[i % len(palette)] for i, m in enumerate(all_models)}
        for h in history:
            c = RED if not h.get("success") else model_color.get(h.get("model", "?"), GRAY)
            ax2.bar(h["num"], 1, color=c, alpha=0.8, width=0.8)
        ax2.set_yticks([]); ax2.set_xlabel("Experiment #")
        ax2.set_title("Models Used", fontsize=11, color=GRAY)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=model_color[m], label=m) for m in all_models if m in model_color]
        if crashes:
            legend_elements.append(Patch(facecolor=RED, label="Crashed"))
        ax2.legend(handles=legend_elements, fontsize=8, framealpha=0.3, ncol=min(4, len(legend_elements)))

        plt.tight_layout()
        p = ws / "experiment_timeline.png"
        plt.savefig(p, dpi=160, facecolor="#0d1117"); plt.close(fig)
        generated["experiment_timeline_png"] = str(p)
    except Exception:
        pass

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
    # model.pkl if saved
    pkl = ws / "model.pkl"
    if pkl.exists(): shutil.copy(pkl, d / "model.pkl")

    inference = extract_code(ask(
        "Write production Python inference code. FastAPI endpoint + CLI. Output ONLY ```python.",
        f"""Production inference script.

MODEL: {best_exp['model']} | TASK: {obj['task']} | TARGET: {obj['target']}
METRIC: {best_exp['metric_name']}={best_exp['metric_val']:.6f}
COLS: {', '.join(profile['headers'])}
NUMERIC: {', '.join(profile['numeric'])}
CATEGORICAL: {', '.join(profile['categorical'])}

Requirements:
- Load model.pkl with joblib
- FastAPI POST /predict — accepts JSON {{"data": [row_dict, ...]}} → returns {{"predictions": [...]}}
- CLI: python predict.py --input data.csv --output predictions.csv
- Same preprocessing as training
- Handle nulls and encoding robustly

Output ONLY ```python code.""", 2500))

    (d / "predict.py").write_text(inference)
    train_code = ""
    if (d / "train.py").exists():
        train_code = (d / "train.py").read_text()
    dep_reqs = build_requirements_from_code(
        train_code,
        inference,
        extra_modules={"fastapi", "uvicorn"},
    )
    (d / "requirements.txt").write_text(dep_reqs)

    history_lines = "\n".join(
        f"- Exp {h['num']:02d}: {h['model']} → {h['metric_name']}={h['metric_val']:.6f}"
        for h in best_exp.get("all_history", []))
    (d / "README.md").write_text(f"""# 19Labs Deployed Model

## Model
- Task: {obj['task']}
- Target: `{obj['target']}`
- Model: **{best_exp['model']}**
- {best_exp['metric_name'].upper()}: **{best_exp['metric_val']:.6f}**
- Trained: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Quick Start
```bash
pip install -r requirements.txt
python train.py        # retrain → saves model.pkl
uvicorn predict:app    # serve API on :8000
```

## API
```bash
POST /predict
{{"data": [{{"col1": val1, "col2": val2}}]}}
→ {{"predictions": [42.0]}}
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
    _init_client(key, provider)
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
            for k in [metric_name, metric_name.upper(), "rmse", "r2", "mape", "mae", "nse", "auc", "f1", "accuracy"]:
                if k in m and _is_valid_metric(k, m[k]):
                    metric_name = k
                    metric_val = float(m[k])
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
                        for k in [metric_name, metric_name.upper(), "rmse", "r2", "mape", "mae", "nse", "auc", "f1", "accuracy"]:
                            if k in m and _is_valid_metric(k, m[k]):
                                metric_name = k
                                metric_val = float(m[k])
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
                        log.result(f"Exp {n}: recovery succeeded → {model_name} {metric_name}={metric_val:.6f}")
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
        revision = revise_after_iteration(program_md, train_py, score, error, history, domain_analysis=domain_analysis)
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
        ge_nums = re.findall(r'[\d,]+\.?\d*', ge_str.replace(',', ''))
        if ge_nums:
            try:
                parsed_t = float(ge_nums[0])
                if parsed_t > 0:
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

    return dict(workspace=str(ws), objective=obj, profile=profile,
        history=history, best=best, deploy_path=deploy, report=report,
        diagnostics=diagnostics, executive_brief=diagnostics.get("executive_brief", ""),
        artifacts=artifacts,
        train_script=str(ws / "train.py"), results_tsv=str(ws / "results.tsv"),
        continuous_mode=continuous, total_experiments=total_experiments,
        token_usage=get_token_usage())

# ── CHAT WITH DATA ─────────────────────────────────────────────
def chat_with_data(message: str, context: dict, api_key: str, provider: str = "claude") -> str:
    """Free-form conversational AI about the current dataset/model."""
    ctx_parts = []
    if context.get("filename"):
        ctx_parts.append(f"Dataset: {context['filename']}")
    profile = context.get("profile") or {}
    if profile.get("n_rows"):
        ctx_parts.append(f"Rows: {profile['n_rows']}, Columns: {profile.get('n_cols', '?')}")
    if profile.get("headers"):
        ctx_parts.append(f"Columns: {', '.join(profile['headers'][:25])}")
    if context.get("best"):
        b = context["best"]
        ctx_parts.append(f"Best model: {b.get('model','?')} — {b.get('metric_name','metric')}: {b.get('metric_val','?')}")
    if context.get("objective"):
        o = context["objective"]
        ctx_parts.append(f"Task: {o.get('task','?')} | Target: {o.get('target','?')} | Metric: {o.get('metric','?')}")
    if context.get("history"):
        kept = sum(1 for e in context["history"] if e.get("success"))
        ctx_parts.append(f"Experiments: {len(context['history'])} total, {kept} kept")

    system = "You are 19Labs, an expert AI data scientist. Answer questions about the dataset, model, results, and data science in general. Be concise, practical, and specific. Use markdown when helpful."
    if ctx_parts:
        system += "\n\nSession context:\n" + "\n".join(f"- {c}" for c in ctx_parts)

    try:
        if provider == "openai" and OpenAI:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
                max_tokens=800
            )
            return resp.choices[0].message.content
        else:
            client = Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": message}]
            )
            return resp.content[0].text
    except Exception as e:
        return f"I couldn't process that: {e}"


# ── INFERENCE SERVER GENERATOR ─────────────────────────────────
def generate_inference_server(train_py: str, best: dict, obj: dict, api_key: str, provider: str = "claude") -> dict:
    """Generate a complete deployable FastAPI inference server from the trained model code."""
    model_name = best.get("model", "MLModel")
    task = obj.get("task", "regression")
    target = obj.get("target", "target")
    metric = obj.get("metric", "accuracy")

    prompt = f"""Generate a production-ready FastAPI inference server for this trained ML model.

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
        if provider == "openai" and OpenAI:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            raw = resp.choices[0].message.content
        else:
            client = Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=CLAUDE_MODEL, max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text

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
