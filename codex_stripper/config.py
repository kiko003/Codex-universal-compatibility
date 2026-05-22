"""Configuration for codex_stripper proxy."""

from __future__ import annotations

import os


class Config:
    """Proxy configuration loaded from environment variables.

    Attributes:
        LISTEN_PORT: Port the proxy listens on.
        UPSTREAM_URL: Base URL of the upstream OpenAI-compatible server.
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        STRIP_STRICT_FIELD: Whether to strip the 'strict' field from tool defs.
    """

    LISTEN_PORT: int
    UPSTREAM_URL: str
    LOG_LEVEL: str
    STRIP_STRICT_FIELD: bool

    def __init__(
        self,
        listen_port: int | None = None,
        upstream_url: str | None = None,
        log_level: str | None = None,
        strip_strict_field: bool | None = None,
    ) -> None:
        self.LISTEN_PORT = listen_port if listen_port is not None else int(
            os.environ.get("STRIPPER_PORT", "8080")
        )
        self.UPSTREAM_URL = upstream_url if upstream_url is not None else os.environ.get(
            "STRIPPER_UPSTREAM", "http://localhost:4000"
        )
        self.LOG_LEVEL = log_level if log_level is not None else os.environ.get(
            "STRIPPER_LOG_LEVEL", "INFO"
        )
        self.STRIP_STRICT_FIELD = strip_strict_field if strip_strict_field is not None else os.environ.get(
            "STRIP_STRICT", "true"
        ).lower() in ("true", "1", "yes")

    def __repr__(self) -> str:
        return (
            f"Config(listen_port={self.LISTEN_PORT}, "
            f"upstream_url={self.UPSTREAM_URL!r}, "
            f"log_level={self.LOG_LEVEL!r}, "
            f"strip_strict_field={self.STRIP_STRICT_FIELD})"
        )
