"""Run TIER CHECK with ``python -m app`` or the ``tier-check`` command."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("TIERCHECK_RELOAD", "").lower() in {"1", "true", "yes"}
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
