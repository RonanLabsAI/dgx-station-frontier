# Derived from original-el8/dgx-station-gb300-research @ 7699ae54, deepseek-v4.1-flash/m3/hook/sitecustomize.py
# (Apache License 2.0, Copyright 2026 Jason Cook), itself modified from J&M Recipes (MIT, Copyright 2026 J&M Recipes;
# licenses/J-M-Recipes-MIT.txt). Modified 2026 by RonanLabs: installs E2's pin-hot hook (when PIN_MODE is set)
# and then the E3 warm-tier hook. Put this directory first on PYTHONPATH (it shadows the image's and E2's
# sitecustomize), so every Python process of the container, vLLM's workers included, runs it.
try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

import os
import sys

if os.environ.get("PIN_MODE", "").strip().lower() in ("split", "off"):
    try:
        import importlib.util as _u
        _s = _u.spec_from_file_location("dsv4pro_pin_hook",
                                        os.environ.get("PIN_HOOK", "/opt/pinhook/dsv4pro_pin_hook.py"))
        _m = _u.module_from_spec(_s)
        sys.modules["dsv4pro_pin_hook"] = _m
        _s.loader.exec_module(_m)
        _m.install()
    except Exception as _e:
        sys.stderr.write(f"PIN_HOT hook FAILED to install: {_e!r}\n")

if os.environ.get("E3_HOOK"):
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("e3_hook", os.environ["E3_HOOK"])
        _m3 = _ilu.module_from_spec(_spec)
        sys.modules["e3_hook"] = _m3
        _spec.loader.exec_module(_m3)
        _m3.install()
    except Exception as _e:
        sys.stderr.write(f"E3 hook FAILED to install: {_e!r}\n")
