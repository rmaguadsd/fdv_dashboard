import json, os, sys
from pathlib import Path
# Ensure the app directory is importable
HERE = Path(__file__).resolve()
APP_DIR = HERE.parent.parent  # d:/dev/aitools
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from fdv_report2_webapp_run import run_filename_resolver_selftest
except Exception as e:
    print(json.dumps({"ok": False, "error": f"import failed: {e}", "sys_path": sys.path}, indent=2))
    sys.exit(2)

def run_with_path(p: str) -> dict:
    from pathlib import Path
    import re
    base_dir = str(Path(p).parent)
    bn = Path(p).name
    stem = Path(p).stem
    # Prefer token captured after tb_set_utility_
    m = re.search(r"(?i)tb_set_utility_([A-Za-z0-9_\-]+)$", stem)
    token = m.group(1) if m else stem
    cases = {}
    # absolute path
    cases['absolute_path'] = {
        'resolved': p,
        'exists': Path(p).is_file(),
        'attempted': []
    }
    # basename under base_dir
    base_candidate = str(Path(base_dir) / bn)
    cases['basename_only'] = {
        'resolved': base_candidate,
        'exists': Path(base_candidate).is_file(),
        'attempted': [base_candidate]
    }
    # token search under base_dir (common patterns)
    attempts = []
    found = None
    if base_dir and token:
        try:
            base = Path(base_dir)
            for pat in [
                f"*_tb_set_utility_{token}.txt",
                f"*_tb_set_utility*{token}*.txt",
                f"*_set_utility_tb_{token}.txt",
                f"*_set_utility_tb*{token}*.txt",
                f"*Output*tb_set_utility*{token}*.txt",
                f"*FDVLOG*{token}*.txt",
                f"*{token}*.txt",
                f"*{token}*.log",
            ]:
                for q in base.rglob(pat):
                    attempts.append(str(q))
                    if q.is_file():
                        found = str(q)
                        break
                if found:
                    break
        except Exception:
            pass
    cases['token_with_dir'] = {
        'resolved': found or token,
        'exists': bool(found and Path(found).is_file()),
        'attempted': attempts
    }
    # token without dir, but app adopts row fdv_file hint: simulate by using p directly
    cases['token_no_dir_but_row_hint'] = {
        'resolved': p,
        'exists': Path(p).is_file(),
        'attempted': []
    }
    return {
        'ok': all(v['exists'] for v in cases.values()),
        'summary': {k: v['exists'] for k, v in cases.items()},
        'cases': cases,
        'base_dir': base_dir,
        'token': token
    }

try:
    if len(sys.argv) > 1:
        res = run_with_path(sys.argv[1])
        print(json.dumps(res, indent=2))
        sys.exit(0 if res.get('ok') else 1)
    else:
        res = run_filename_resolver_selftest()
        print(json.dumps(res, indent=2))
        sys.exit(0 if res.get('ok') else 1)
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}, indent=2))
    sys.exit(3)
