from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_login_page_does_not_load_auth_backend_before_user_action() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ui.auth_page; "
                "print(any(name == 'src.auth' or name.startswith('src.auth.') "
                "for name in sys.modules))"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
