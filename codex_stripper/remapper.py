"""Response-side namespace remapping for Codex MCP tool dispatch.

When an upstream provider returns a function_call with a flattened tool name
(e.g. mcp__context7__Context7_query_docs), this module remaps the response
back so Codex can dispatch MCP tools correctly.

For /v1/responses format the flattened name is kept as-is (Codex dispatches
by matching the mcp__{server}__{tool_name} pattern), but the name is validated
against the namespace_map.

For /v1/chat/completions format the flattened name is remapped back to the
original_name stored in the namespace_map.
"""

from __future__ import annotations

import copy


def remap_function_call_name(flattened_name: str, namespace_map: dict) -> str:
    """Return the original tool name for a flattened name, or pass through unchanged.

    Args:
        flattened_name: The flattened function name (e.g. mcp__context7__Context7_query_docs).
        namespace_map: Mapping of flattened_name -> {"original_name": ..., **extra}.
            Only the "original_name" key is consulted.

    Returns:
        The original un-flattened name if *flattened_name* exists in *namespace_map*,
        otherwise *flattened_name* unchanged.
    """
    entry = namespace_map.get(flattened_name)
    if entry is not None and isinstance(entry, dict):
        original = entry.get("original_name")
        if original is not None:
            return original
    return flattened_name


def remap_response_body(body: dict, namespace_map: dict, endpoint: str = "responses") -> dict:
    """Remap function_call names in a response body for the given endpoint.

    Args:
        body: The deserialized JSON response body.
        namespace_map: Mapping of flattened_name -> {"original_name": ..., **extra}.
        endpoint: Either ``"responses"`` or ``"chat/completions"``.

    Returns:
        A **new** dict with the remapping applied.  The original *body* is not mutated.

    Behaviour:
    * **responses** – Names inside ``output[].function_call.name`` are kept as-is
      (Codex uses the flattened name for MCP dispatch) but every name is validated
      to exist in *namespace_map* (a ``KeyError`` is raised if a flattened name is
      missing).
    * **chat/completions** – Every ``choices[].message.tool_calls[].function.name``
      is remapped back to the original name via :func:`remap_function_call_name`.
    """
    result = copy.deepcopy(body)

    if endpoint == "responses":
        for item in result.get("output", []):
            fc = item.get("function_call")
            if fc is None:
                continue
            name = fc.get("name")
            if name is not None and name not in namespace_map:
                raise KeyError(
                    f"Flattened function name '{name}' not found in namespace_map"
                )
        return result

    if endpoint == "chat/completions":
        for choice in result.get("choices", []):
            msg = choice.get("message", {})
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                if "name" in fn:
                    fn["name"] = remap_function_call_name(fn["name"], namespace_map)
        return result

    return result
