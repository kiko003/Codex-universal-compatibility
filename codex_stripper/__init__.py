"""codex_stripper: transforms Codex namespace-wrapped MCP tools into flat function tools."""

from .transformer import flatten_namespace_tools, flatten_request_body

__all__ = ["flatten_namespace_tools", "flatten_request_body"]
