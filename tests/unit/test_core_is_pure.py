"""Guard: openbb_chokepoints.core must not import anything from OpenBB."""

import pkgutil
import subprocess
import sys

import openbb_chokepoints.core as core


def _core_modules() -> list[str]:
    return [core.__name__] + [
        f"{core.__name__}.{m.name}" for m in pkgutil.iter_modules(core.__path__)
    ]


def test_core_does_not_import_openbb():
    imports = "; ".join(f"import {name}" for name in _core_modules())
    code = (
        f"import sys; {imports}; "
        "bad = sorted(m for m in sys.modules "
        "if m.startswith('openbb') and not m.startswith('openbb_chokepoints')); "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""
