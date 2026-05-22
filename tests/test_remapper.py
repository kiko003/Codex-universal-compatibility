"""Unit tests for codex_stripper.remapper."""

import pytest

from codex_stripper.remapper import remap_function_call_name, remap_response_body


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

NAMESPACE_MAP = {
    "mcp__context7__Context7_query_docs": {
        "original_name": "Context7_query_docs",
        "server": "context7",
    },
    "mcp__homelab__read_file": {
        "original_name": "read_file",
        "server": "homelab",
    },
}


# ---------------------------------------------------------------------------
# remap_function_call_name
# ---------------------------------------------------------------------------

class TestRemapFunctionCallName:
    def test_remap_known_flattened_name(self):
        assert remap_function_call_name(
            "mcp__context7__Context7_query_docs", NAMESPACE_MAP
        ) == "Context7_query_docs"

    def test_pass_through_unknown_name(self):
        assert remap_function_call_name(
            "unknown_function", NAMESPACE_MAP
        ) == "unknown_function"

    def test_empty_namespace_map(self):
        assert remap_function_call_name(
            "mcp__context7__Context7_query_docs", {}
        ) == "mcp__context7__Context7_query_docs"


# ---------------------------------------------------------------------------
# remap_response_body – /v1/chat/completions
# ---------------------------------------------------------------------------

class TestRemapResponseBodyChatCompletions:
    def test_full_chat_completions_response(self):
        body = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "mcp__context7__Context7_query_docs",
                                    "arguments": '{"libraryId": "/mongodb/docs"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = remap_response_body(body, NAMESPACE_MAP, endpoint="chat/completions")
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "Context7_query_docs"

    def test_multiple_tool_calls(self):
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "mcp__context7__Context7_query_docs",
                                    "arguments": "{}",
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "mcp__homelab__read_file",
                                    "arguments": "{}",
                                },
                            },
                            {
                                "id": "call_3",
                                "type": "function",
                                "function": {
                                    "name": "unknown_tool",
                                    "arguments": "{}",
                                },
                            },
                        ]
                    }
                }
            ]
        }
        result = remap_response_body(body, NAMESPACE_MAP, endpoint="chat/completions")
        calls = result["choices"][0]["message"]["tool_calls"]
        assert calls[0]["function"]["name"] == "Context7_query_docs"
        assert calls[1]["function"]["name"] == "read_file"
        assert calls[2]["function"]["name"] == "unknown_tool"

    def test_empty_namespace_map_passthrough(self):
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "mcp__context7__Context7_query_docs",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        result = remap_response_body(body, {}, endpoint="chat/completions")
        assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == \
            "mcp__context7__Context7_query_docs"

    def test_does_not_mutate_original(self):
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "mcp__context7__Context7_query_docs",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        remap_response_body(body, NAMESPACE_MAP, endpoint="chat/completions")
        assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == \
            "mcp__context7__Context7_query_docs"


# ---------------------------------------------------------------------------
# remap_response_body – /v1/responses
# ---------------------------------------------------------------------------

class TestRemapResponseBodyResponses:
    def test_responses_name_kept_as_is(self):
        body = {
            "id": "resp-123",
            "output": [
                {
                    "type": "function_call",
                    "function_call": {
                        "name": "mcp__context7__Context7_query_docs",
                        "arguments": '{"libraryId": "/mongodb/docs"}',
                    },
                }
            ],
        }
        result = remap_response_body(body, NAMESPACE_MAP, endpoint="responses")
        assert result["output"][0]["function_call"]["name"] == \
            "mcp__context7__Context7_query_docs"

    def test_responses_unknown_name_raises(self):
        body = {
            "output": [
                {
                    "type": "function_call",
                    "function_call": {
                        "name": "unknown_function",
                        "arguments": "{}",
                    },
                }
            ],
        }
        with pytest.raises(KeyError, match="unknown_function"):
            remap_response_body(body, NAMESPACE_MAP, endpoint="responses")

    def test_responses_empty_namespace_map_raises(self):
        body = {
            "output": [
                {
                    "function_call": {
                        "name": "mcp__context7__Context7_query_docs",
                        "arguments": "{}",
                    },
                }
            ],
        }
        with pytest.raises(KeyError):
            remap_response_body(body, {}, endpoint="responses")

    def test_responses_no_function_call_items(self):
        body = {
            "output": [
                {"type": "message", "content": "hello"},
            ]
        }
        result = remap_response_body(body, NAMESPACE_MAP, endpoint="responses")
        assert result == body

    def test_does_not_mutate_original(self):
        body = {
            "output": [
                {
                    "function_call": {
                        "name": "mcp__context7__Context7_query_docs",
                        "arguments": "{}",
                    },
                }
            ],
        }
        original_output = body["output"]
        remap_response_body(body, NAMESPACE_MAP, endpoint="responses")
        assert body["output"] is original_output


# ---------------------------------------------------------------------------
# remap_response_body – unknown endpoint
# ---------------------------------------------------------------------------

class TestRemapResponseBodyUnknownEndpoint:
    def test_unknown_endpoint_returns_copy_unchanged(self):
        body = {"id": "x", "output": []}
        result = remap_response_body(body, NAMESPACE_MAP, endpoint="unknown")
        assert result == body
        assert result is not body
