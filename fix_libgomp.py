#!/usr/bin/env python3
"""
Run after `pip install -r requirements.txt` in the build phase.
Finds libgomp.so.1 (bundled inside xgboost/lightgbm pip wheels or system)
and copies it to /usr/local/lib/libgomp.so.1 so every process can find it.
"""
import glob, os, shutil, sys

DST = "/usr/local/lib/libgomp.so.1"

def find_candidates():
    candidates = []
    # 1. xgboost/lightgbm wheel-bundled paths
    for sp in sys.path:
        candidates += glob.glob(sp + "/xgboost.libs/libgomp*")
        candidates += glob.glob(sp + "/lightgbm.libs/libgomp*")
        candidates += glob.glob(sp + "/*.libs/libgomp*")
    # 2. xgboost package directory itself
    try:
        import importlib.util
        spec = importlib.util.find_spec("xgboost")
        if spec and spec.origin:
            xgb_dir = os.path.dirname(spec.origin)
            candidates += glob.glob(xgb_dir + "/../xgboost.libs/libgomp*")
            candidates += glob.glob(xgb_dir + ".libs/libgomp*")
    except Exception as e:
        print(f"  xgboost spec lookup: {e}")
    # 3. System paths
    candidates += glob.glob("/usr/lib/x86_64-linux-gnu/libgomp.so*")
    candidates += glob.glob("/usr/lib/aarch64-linux-gnu/libgomp.so*")
    candidates += glob.glob("/usr/lib/*/libgomp.so*")
    candidates += glob.glob("/usr/local/lib/libgomp.so*")
    candidates += glob.glob("/usr/lib/libgomp.so*")
    # 4. Nix store
    candidates += glob.glob("/nix/store/*/lib/libgomp.so*")
    return [c for c in candidates if os.path.isfile(c)]

print("=== fix_libgomp.py ===")
candidates = find_candidates()
print(f"Candidates found: {candidates}")

if not candidates:
    print("ERROR: No libgomp found anywhere — prediction of xgb/lgbm models will fail")
    sys.exit(0)  # don't fail the build

src = candidates[0]
print(f"Using: {src}")
os.makedirs("/usr/local/lib", exist_ok=True)

if os.path.exists(DST):
    os.remove(DST)

shutil.copy2(src, DST)
print(f"Copied to {DST}")

# Patch the ELF SONAME so the linker registers it as "libgomp.so.1".
# Without this, loading the file still registers under the mangled name
# (e.g. libgomp-e985bcbb.so.1.0.0) and lightgbm's dlopen("libgomp.so.1") fails.
try:
    import subprocess
    result = subprocess.run(
        ["patchelf", "--set-soname", "libgomp.so.1", DST],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("patchelf: SONAME set to libgomp.so.1")
    else:
        print(f"patchelf failed (non-fatal): {result.stderr.strip()}")
except FileNotFoundError:
    print("patchelf not found — SONAME not patched (ldconfig may not map libgomp.so.1 correctly)")
except Exception as e:
    print(f"patchelf error (non-fatal): {e}")

# Also create symlink for any versioned name differences
for alias in ["/usr/local/lib/libgomp.so", "/usr/local/lib/libgomp.so.1.0.0"]:
    if not os.path.exists(alias):
        try:
            os.symlink(DST, alias)
            print(f"Symlinked {alias}")
        except Exception as e:
            print(f"  symlink {alias}: {e}")

print("Done.")
