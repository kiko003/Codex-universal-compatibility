"""Unit tests for codex_stripper.transformer.

Covers:
- Single namespace with one tool
- Single namespace with multiple tools
- Mixed: namespace tools + flat function tools
- /v1/chat/completions format (tools[].function)
- Empty tools array
- No tools key at all
- Nested namespace edge case
- Strict field stripping
"""

from __future__ import annotations

import pytest

from codex_stripper.transformer import flatten_namespace_tools, flatten_request_body


# -----------------------------------------------------------------------
# Fixtures / helpers
# -----------------------------------------------------------------------

def _ns_tool(name: str, sub_tools: list[dict]) -> dict:
    """Build a namespace tool dict."""
    return {"type": "namespace", "name": name, "tools": sub_tools}


def _func_tool(name: str, description: str = "desc", strict: bool = True, **extra) -> dict:
    """Build a function tool dict (responses format)."""
    t: dict = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
        "strict": strict,
    }
    t.update(extra)
    return t


# -----------------------------------------------------------------------
# flatten_namespace_tools
# -----------------------------------------------------------------------

class TestFlattenNamespaceTools:
    """Tests for flatten_namespace_tools."""

    def test_single_namespace_one_tool(self):
        ns = _ns_tool("mcp__context7__", [
            _func_tool("Context7_query_docs", "Query docs"),
        ])
        flat, ns_map = flatten_namespace_tools([ns])

        assert len(flat) == 1
        assert flat[0]["type"] == "function"
        assert flat[0]["name"] == "mcp__context7__Context7_query_docs"
        assert "strict" not in flat[0]
        assert "mcp__context7__Context7_query_docs" in ns_map
        assert ns_map["mcp__context7__Context7_query_docs"]["namespace"] == "mcp__context7__"
        assert ns_map["mcp__context7__Context7_query_docs"]["original_name"] == "Context7_query_docs"

    def test_single_namespace_multiple_tools(self):
        ns = _ns_tool("mcp__context7__", [
            _func_tool("Context7_query_docs", "Query docs"),
            _func_tool("Context7_resolve_library_id", "Resolve library ID"),
        ])
        flat, ns_map = flatten_namespace_tools([ns])

        assert len(flat) == 2
        assert flat[0]["name"] == "mcp__context7__Context7_query_docs"
        assert flat[1]["name"] == "mcp__context7__Context7_resolve_library_id"
        assert len(ns_map) == 2

    def test_mixed_namespace_and_flat_tools(self):
        ns = _ns_tool("mcp__homelab__", [
            _func_tool("Homelab_read_file", "Read file"),
        ])
        plain = _func_tool("web_search", "Search the web")

        flat, ns_map = flatten_namespace_tools([ns, plain])

        assert len(flat) == 2
        assert flat[0]["name"] == "mcp__homelab__Homelab_read_file"
        assert flat[1]["name"] == "web_search"
        assert "web_search" not in ns_map
        assert "mcp__homelab__Homelab_read_file" in ns_map

    def test_strict_stripped_from_flat_function(self):
        plain = _func_tool("my_tool", "desc", strict=True)
        flat, _ = flatten_namespace_tools([plain])

        assert "strict" not in flat[0]

    def test_non_function_type_passthrough(self):
        other = {"type": "computer_20241022", "display_width": 1280}
        flat, ns_map = flatten_namespace_tools([other])

        assert len(flat) == 1
        assert flat[0]["type"] == "computer_20241022"
        assert ns_map == {}

    def test_empty_tools_array(self):
        flat, ns_map = flatten_namespace_tools([])
        assert flat == []
        assert ns_map == {}

    def test_nested_namespace(self):
        """Namespace inside a namespace: inner tools get the combined prefix."""
        inner = _ns_tool("inner__", [
            _func_tool("InnerTool", "Does a thing"),
        ])
        outer = _ns_tool("outer__", [inner])

        flat, ns_map = flatten_namespace_tools([outer])

        assert len(flat) == 1
        assert flat[0]["name"] == "outer__inner__InnerTool"
        assert ns_map["outer__inner__InnerTool"]["namespace"] == "outer__inner__"
        assert ns_map["outer__inner__InnerTool"]["original_name"] == "InnerTool"

    def test_description_combines_namespace(self):
        ns = _ns_tool("mcp__context7__", [
            _func_tool("Context7_query_docs", "Query documentation"),
        ])
        flat, _ = flatten_namespace_tools([ns])

        desc = flat[0]["description"]
        assert "[mcp / context7]" in desc
        assert "Query documentation" in desc

    def test_input_not_mutated(self):
        """Ensure the original tools list is not modified."""
        original_tool = _func_tool("keep_me", "desc")
        original_copy = dict(original_tool)
        _ = flatten_namespace_tools([original_tool])
        assert original_tool == original_copy


# -----------------------------------------------------------------------
# flatten_request_body – /v1/responses format
# -----------------------------------------------------------------------

class TestFlattenRequestBodyResponses:
    """Tests for flatten_request_body with /v1/responses format."""

    def test_responses_format(self):
        body = {
            "model": "gpt-4o",
            "tools": [
                _ns_tool("mcp__ctx7__", [
                    _func_tool("Ctx7_query", "Query"),
                ]),
            ],
        }
        result, ns_map = flatten_request_body(body)

        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "mcp__ctx7__Ctx7_query"
        assert result["model"] == "gpt-4o"
        assert "mcp__ctx7__Ctx7_query" in ns_map

    def test_no_tools_key(self):
        body = {"model": "gpt-4o", "input": "hello"}
        result, ns_map = flatten_request_body(body)

        assert "tools" not in result or result.get("tools") is None
        assert ns_map == {}

    def test_empty_tools(self):
        body = {"model": "gpt-4o", "tools": []}
        result, ns_map = flatten_request_body(body)

        assert result["tools"] == []
        assert ns_map == {}

    def test_original_body_not_mutated(self):
        body = {
            "model": "gpt-4o",
            "tools": [_ns_tool("ns__", [_func_tool("T", "d")])],
        }
        original_tools_len = len(body["tools"])
        _ = flatten_request_body(body)
        assert len(body["tools"]) == original_tools_len


# -----------------------------------------------------------------------
# flatten_request_body – /v1/chat/completions format
# -----------------------------------------------------------------------

class TestFlattenRequestBodyChatCompletions:
    """Tests for flatten_request_body with /v1/chat/completions format."""

    def test_chat_completions_format(self):
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    },
                },
            ],
        }
        result, ns_map = flatten_request_body(body)

        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "web_search"
        assert "strict" not in tool["function"]
        assert ns_map == {}

    def test_chat_completions_with_namespace(self):
        body = {
            "model": "gpt-4o",
            "messages": [],
            "tools": [
                _ns_tool("mcp__homelab__", [
                    _func_tool("Homelab_read_file", "Read a file"),
                ]),
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }
        result, ns_map = flatten_request_body(body)

        assert len(result["tools"]) == 2
        # The namespace tool should be flattened into function format
        ns_tool_out = result["tools"][0]
        assert ns_tool_out["type"] == "function"
        assert ns_tool_out["function"]["name"] == "mcp__homelab__Homelab_read_file"
        # The plain function should stay but be re-wrapped
        plain_out = result["tools"][1]
        assert plain_out["function"]["name"] == "get_weather"
        assert "mcp__homelab__Homelab_read_file" in ns_map

    def test_chat_completions_mixed(self):
        body = {
            "model": "deepseek-chat",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "plain_tool",
                        "description": "A plain tool",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    },
                },
                _ns_tool("mcp__db__", [
                    _func_tool("DB_query", "Run a query"),
                    _func_tool("DB_insert", "Insert a row"),
                ]),
            ],
        }
        result, ns_map = flatten_request_body(body)

        assert len(result["tools"]) == 3
        names = [t["function"]["name"] for t in result["tools"]]
        assert "plain_tool" in names
        assert "mcp__db__DB_query" in names
        assert "mcp__db__DB_insert" in names
        assert "strict" not in result["tools"][0]["function"]
        assert len(ns_map) == 2


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Misc edge-case tests."""

    def test_namespace_with_empty_sub_tools(self):
        ns = _ns_tool("empty_ns__", [])
        flat, ns_map = flatten_namespace_tools([ns])

        assert flat == []
        assert ns_map == {}

    def test_namespace_with_no_name(self):
        ns = _ns_tool("", [_func_tool("orphan_tool", "desc")])
        flat, ns_map = flatten_namespace_tools([ns])

        assert len(flat) == 1
        assert flat[0]["name"] == "orphan_tool"

    def test_function_with_extra_fields(self):
        """Extra fields on a function tool are preserved."""
        tool = _func_tool("extra_tool", "desc", custom_field="keep_me")
        flat, _ = flatten_namespace_tools([tool])

        assert flat[0]["custom_field"] == "keep_me"

    def test_deeply_nested_namespaces(self):
        """Three levels of nesting."""
        innermost = _ns_tool("c__", [_func_tool("Deep", "Deepest tool")])
        mid = _ns_tool("b__", [innermost])
        outer = _ns_tool("a__", [mid])

        flat, ns_map = flatten_namespace_tools([outer])

        assert len(flat) == 1
        assert flat[0]["name"] == "a__b__c__Deep"
        assert ns_map["a__b__c__Deep"]["namespace"] == "a__b__c__"
