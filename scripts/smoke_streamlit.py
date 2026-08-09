"""Start the real Streamlit server and verify its HTTP health endpoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALTH_URL = "http://127.0.0.1:8517/_stcore/health"
STARTUP_TIMEOUT_SECONDS = 45.0


def _server_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/streamlit_app.py",
        "--server.headless=true",
        "--server.address=127.0.0.1",
        "--server.port=8517",
        "--browser.gatherUsageStats=false",
    ]


def main() -> int:
    """Return zero only when a real Streamlit process becomes healthy."""
    environment = os.environ.copy()
    environment.setdefault("QUANT_SIM_ENV", "test")

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            _server_command(),
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    with urlopen(HEALTH_URL, timeout=1) as response:
                        if response.status == 200 and response.read().strip() == b"ok":
                            return 0
                except (OSError, URLError):
                    time.sleep(0.25)

            log_file.seek(0)
            sys.stderr.write(log_file.read())
            return 1
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
