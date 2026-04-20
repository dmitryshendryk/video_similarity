"""CLI entry point for the FastAPI server (used by `poetry run s2vs-server`).

Spawns uvicorn as a subprocess so that KMP_DUPLICATE_LIB_OK is set in the
process environment *before* any C library (OpenMP/MKL via torch/faiss) loads.
Setting os.environ inside the same process is too late on macOS.
"""

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="S2VS video deduplication API server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api_server:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--loop",
        "asyncio",
    ]
    if args.reload:
        cmd.append("--reload")

    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
