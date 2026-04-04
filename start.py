from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "oceanpulse-dev.log"


def _stream_output(name: str, pipe, log_file):
    for line in iter(pipe.readline, ""):
        message = f"[{name}] {line}"
        print(message, end="")
        log_file.write(message)
        log_file.flush()
    pipe.close()


def _backend_command() -> list[str]:
    candidates = [
        BACKEND_DIR / ".venv" / "bin" / "python",
        BACKEND_DIR / ".venv" / "Scripts" / "python.exe",
    ]
    python_bin = next((candidate for candidate in candidates if candidate.exists()), None)
    if python_bin is None:
        raise FileNotFoundError(
            "Backend virtualenv not found. Create it first with `cd backend && python3.11 -m venv .venv`."
        )
    return [str(python_bin), "-m", "uvicorn", "main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"]


def _frontend_command() -> list[str]:
    npm = "npm.cmd" if os.name == "nt" else "npm"
    return [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]


def main():
    LOG_DIR.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== OceanPulse dev session {datetime.now().isoformat()} ===\n")
        log_file.flush()

        processes = [
            (
                "backend",
                subprocess.Popen(
                    _backend_command(),
                    cwd=BACKEND_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ),
            ),
            (
                "frontend",
                subprocess.Popen(
                    _frontend_command(),
                    cwd=FRONTEND_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ),
            ),
        ]

        threads = [
            threading.Thread(target=_stream_output, args=(name, process.stdout, log_file), daemon=True)
            for name, process in processes
        ]
        for thread in threads:
            thread.start()

        def shutdown(*_args):
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        exit_code = 0
        try:
            for name, process in processes:
                code = process.wait()
                if code != 0:
                    print(f"[starter] {name} exited with code {code}")
                    exit_code = code or 1
                    shutdown()
                    break
        finally:
            shutdown()
            for _, process in processes:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            for thread in threads:
                thread.join(timeout=1)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
