"""Configuration for codex_stripper proxy."""

from __future__ import annotations

import os


class Config:
    """Proxy configuration loaded from environment variables.

    Attributes:
        LISTEN_PORT: Port the proxy listens on.
        UPSTREAM_URL: Base URL of the upstream OpenAI-compatible server.
        UPSTREAM_API_KEY: Bearer token for authenticating with the upstream.
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        STRIP_STRICT_FIELD: Whether to strip the 'strict' field from tool defs.
    """

    LISTEN_PORT: int
    UPSTREAM_URL: str
    UPSTREAM_API_KEY: str
    LOG_LEVEL: str
    STRIP_STRICT_FIELD: bool
    STRIP_NON_FUNCTION_TOOLS: bool

    def __init__(
        self,
        listen_port: int | None = None,
        upstream_url: str | None = None,
        upstream_api_key: str | None = None,
        log_level: str | None = None,
        strip_strict_field: bool | None = None,
        strip_non_function_tools: bool | None = None,
    ) -> None:
        self.LISTEN_PORT = listen_port if listen_port is not None else int(
            os.environ.get("STRIPPER_PORT", "8080")
        )
        self.UPSTREAM_URL = upstream_url if upstream_url is not None else os.environ.get(
            "STRIPPER_UPSTREAM", "http://localhost:4000"
        )
        self.UPSTREAM_API_KEY = upstream_api_key if upstream_api_key is not None else os.environ.get(
            "UPSTREAM_API_KEY", ""
        )
        self.LOG_LEVEL = log_level if log_level is not None else os.environ.get(
            "STRIPPER_LOG_LEVEL", "INFO"
        )
        self.STRIP_STRICT_FIELD = strip_strict_field if strip_strict_field is not None else os.environ.get(
            "STRIP_STRICT", "true"
        ).lower() in ("true", "1", "yes")
        self.STRIP_NON_FUNCTION_TOOLS = strip_non_function_tools if strip_non_function_tools is not None else os.environ.get(
            "STRIP_NON_FUNCTION", "true"
        ).lower() in ("true", "1", "yes")

    def __repr__(self) -> str:
        return (
            f"Config(listen_port={self.LISTEN_PORT}, "
            f"upstream_url={self.UPSTREAM_URL!r}, "
            f"upstream_api_key={'***' if self.UPSTREAM_API_KEY else ''}, "
            f"log_level={self.LOG_LEVEL!r}, "
            f"strip_strict_field={self.STRIP_STRICT_FIELD}, "
            f"strip_non_function_tools={self.STRIP_NON_FUNCTION_TOOLS})"
        )
