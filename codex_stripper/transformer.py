"""Transformer module for flattening Codex namespace-wrapped MCP tools.

Codex (and OpenAI's /v1/responses endpoint) wraps MCP tools inside a
``type=namespace`` container.  Many OpenAI-compatible providers (NVIDIA NIM,
DeepSeek, etc.) only understand flat ``type=function`` tools.  This module
recursively flattens the namespace hierarchy so that every sub-tool becomes a
stand-alone function tool with a dotted name (e.g. ``mcp__context7__Context7_query_docs``).

Public API
----------
- :func:`flatten_namespace_tools`  – flatten a raw ``tools`` list.
- :func:`flatten_request_body`     – flatten a full request body (supports
  both ``/v1/responses`` and ``/v1/chat/completions`` formats).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# ---------------------------------------------------------------------------
# Core flattening
# ---------------------------------------------------------------------------

def flatten_namespace_tools(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Flatten namespace-wrapped MCP tools into flat function tools.

    Parameters
    ----------
    tools:
        The ``tools`` array as it appears in a Codex request body.  Each entry
        is a dict that may be ``{"type": "namespace", ...}`` or
        ``{"type": "function", ...}`` (or any other type).

    Returns
    -------
    flattened_tools : list[dict[str, Any]]
        A new list where every namespace has been expanded into its constituent
        function tools.  Plain ``type=function`` tools (and any unrecognised
        types) are passed through unchanged.

    namespace_map : dict[str, dict[str, str]]
        A mapping from each flattened function name to its lineage::

            {
                "mcp__context7__Context7_query_docs": {
                    "namespace": "mcp__context7__",
                    "original_name": "Context7_query_docs",
                },
                ...
            }

        Tools that were *not* inside a namespace are absent from this map.

    Notes
    -----
    - The ``"strict"`` field is stripped from every flattened function tool
      because many upstream providers reject it.
    - Nested namespaces (a namespace containing a namespace) are handled by
      prepending each level of namespace to the inner tool names.  The
      inner namespace itself is treated as a pass-through container (its
      sub-tools are extracted recursively).
    - The input list is **not** mutated; deep copies are used throughout.
    """
    flattened: list[dict[str, Any]] = []
    namespace_map: dict[str, dict[str, str]] = {}

    for tool in tools:
        tool_type = tool.get("type")

        if tool_type == "namespace":
            _flatten_namespace(tool, prefix="", namespace_map=namespace_map, out=flattened)
        else:
            # Pass through unchanged (deep-copied so caller can't mutate input).
            entry = deepcopy(tool)
            if tool_type == "function":
                _strip_strict(entry)
            flattened.append(entry)

    return flattened, namespace_map


# ---------------------------------------------------------------------------
# Request-body flattening
# ---------------------------------------------------------------------------

def flatten_request_body(
    body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Flatten namespace tools in a full API request body.

    Supports two OpenAI-compatible formats:

    * **``/v1/responses``** – tools live at ``body["tools"]`` and each tool is
      a top-level dict with ``"type"`` and (for functions) ``"name"``,
      ``"description"``, ``"parameters"``.
    * **``/v1/chat/completions``** – tools live at ``body["tools"]`` but each
      entry is ``{"type": "function", "function": {"name", ...}}``.

    If the body has no ``"tools"`` key, or the value is an empty list, the body
    is returned unchanged (with an empty namespace map).

    Parameters
    ----------
    body:
        The full request body dict.

    Returns
    -------
    transformed_body : dict[str, Any]
        A **new** dict (the original is not mutated) with the ``tools`` array
        replaced by its flattened version.

    namespace_map : dict[str, dict[str, str]]
        See :func:`flatten_namespace_tools`.
    """
    result = deepcopy(body)
    tools = result.get("tools")

    if not tools:
        return result, {}

    # Detect chat/completions format: each tool has a "function" key.
    is_chat_completions = any("function" in t for t in tools)

    if is_chat_completions:
        flat_tools, ns_map = _flatten_chat_completions_tools(tools)
    else:
        flat_tools, ns_map = flatten_namespace_tools(tools)

    result["tools"] = flat_tools
    return result, ns_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_namespace(
    ns_tool: dict[str, Any],
    *,
    prefix: str,
    namespace_map: dict[str, dict[str, str]],
    out: list[dict[str, Any]],
) -> None:
    """Recursively flatten a single namespace tool into *out*.

    *prefix* is the dot-separated namespace path built so far (e.g.
    ``"mcp__context7__"``).
    """
    ns_name: str = ns_tool.get("name", "")
    new_prefix = f"{prefix}{ns_name}" if prefix else ns_name

    sub_tools: list[dict[str, Any]] = ns_tool.get("tools", [])

    for sub in sub_tools:
        sub_type = sub.get("type")

        if sub_type == "namespace":
            # Nested namespace – recurse with accumulated prefix.
            _flatten_namespace(sub, prefix=new_prefix, namespace_map=namespace_map, out=out)
        elif sub_type == "function":
            flat = _flatten_function(sub, prefix=new_prefix)
            flat_name = flat["name"]
            namespace_map[flat_name] = {
                "namespace": new_prefix,
                "original_name": sub.get("name", ""),
            }
            out.append({"type": "function", **flat})
        else:
            # Unknown sub-tool type – pass through as-is.
            out.append(deepcopy(sub))


def _flatten_function(func_tool: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    """Build a flat function dict with the namespace prepended to the name."""
    name = func_tool.get("name", "")
    flat_name = f"{prefix}{name}"

    flat: dict[str, Any] = {
        "name": flat_name,
        "description": _combine_description(prefix, func_tool.get("description", "")),
        "parameters": deepcopy(func_tool.get("parameters", {})),
    }

    return flat


def _combine_description(namespace: str, desc: str) -> str:
    """Combine namespace context with the tool's own description."""
    ns_label = namespace.rstrip("_").replace("__", " / ").replace("_", " ")
    if desc:
        return f"[{ns_label}] {desc}"
    return f"[{ns_label}]"


def _strip_strict(tool: dict[str, Any]) -> None:
    """Remove the ``strict`` key from a function tool dict (in-place)."""
    tool.pop("strict", None)


def _flatten_chat_completions_tools(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Flatten tools in ``/v1/chat/completions`` wrapping.

    Each tool is ``{"type": "function", "function": {"name", ...}}``.
    Namespaces use the same wrapping but ``type`` is ``"namespace"``.
    """
    # Normalise to the /v1/responses flat structure, flatten, then re-wrap.
    normalised: list[dict[str, Any]] = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            normalised.append({"type": "function", **deepcopy(t["function"])})
        else:
            normalised.append(deepcopy(t))

    flat, ns_map = flatten_namespace_tools(normalised)

    # Re-wrap each function tool back into {"type": "function", "function": {...}}.
    rewrapped: list[dict[str, Any]] = []
    for ft in flat:
        if ft.get("type") == "function":
            func_dict = {k: v for k, v in ft.items() if k != "type"}
            rewrapped.append({"type": "function", "function": func_dict})
        else:
            rewrapped.append(ft)

    return rewrapped, ns_map
