"""`regintel` console entry point: serve the API with uvicorn."""

from __future__ import annotations

import os

import uvicorn


def start() -> None:
    from regintel.runtime import use_os_certificates

    use_os_certificates()
    uvicorn.run(
        "regintel.api.app:create_app",
        factory=True,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    start()
