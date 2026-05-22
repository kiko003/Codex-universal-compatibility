"""codex_stripper: transforms Codex namespace-wrapped MCP tools into flat function tools."""

from .transformer import flatten_namespace_tools, flatten_request_body
from .remapper import remap_function_call_name, remap_response_body
from .config import Config

__all__ = [
    "flatten_namespace_tools",
    "flatten_request_body",
    "remap_function_call_name",
    "remap_response_body",
    "Config",
]
