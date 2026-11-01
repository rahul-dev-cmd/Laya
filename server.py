#!/usr/bin/env python3
"""Local development server for the Laya speech coaching dashboard.

Runs the FastAPI backend application via uvicorn, serving the frontend
interface and all API endpoints.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer project's virtual environment when running under another Python executable
venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if venv_python.exists() and sys.executable.lower() != str(venv_python).lower():
    import subprocess
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0:
        sys.exit(subprocess.call([str(venv_python)] + sys.argv))

import uvicorn

PORT = 8000


def run_server(port: int = PORT):
    """Run local FastAPI development server via uvicorn."""
    print(f"[Laya Server] Running at http://localhost:{port}/ and http://127.0.0.1:{port}/")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", PORT))
    run_server(port_arg)
