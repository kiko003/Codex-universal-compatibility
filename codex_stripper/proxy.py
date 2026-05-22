"""Aiohttp-based reverse proxy for OpenAI Codex CLI tool transformation.

Wires the transformer (request-side namespace flattening) and remapper
(response-side namespace restoration) into the proxy pipeline.
"""

import json
import logging

from aiohttp import web, ClientSession

from .transformer import flatten_request_body
from .remapper import remap_response_body
from .config import Config

logger = logging.getLogger("codex_stripper.proxy")

# Routes that need tool namespace transformation
TRANSFORM_ROUTES = {"/v1/responses", "/v1/chat/completions"}

# Per-request namespace map storage: keyed by aiohttp request id
_namespace_maps: dict[int, dict] = {}


# ---------------------------------------------------------------------------
# Transformation hooks (wired to real transformer/remapper)
# ---------------------------------------------------------------------------

async def transform_tools_request(body: dict) -> tuple[dict, dict]:
    """Flatten namespace-wrapped tools in the request body.

    Returns:
        (transformed_body, namespace_map) -- namespace_map is stored
        per-request so the response hook can un-flatten.
    """
    transformed, namespace_map = flatten_request_body(body)

    if namespace_map:
        namespaces = {v["namespace"] for v in namespace_map.values()}
        logger.info(
            "Flattened %d namespace tools from %d namespace(s): %s",
            len(namespace_map),
            len(namespaces),
            ", ".join(sorted(namespaces)),
        )

    return transformed, namespace_map


async def transform_tools_response(
    body: dict, namespace_map: dict, *, endpoint: str = "responses",
) -> dict:
    """Remap flattened function names back in the response body."""
    if not namespace_map:
        return body
    try:
        return remap_response_body(body, namespace_map, endpoint=endpoint)
    except KeyError as exc:
        logger.warning(
            "Response remapping failed for a function call: %s. "
            "Returning response unmodified.",
            exc,
        )
        return body


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forward_headers(request: web.Request) -> dict:
    """Copy request headers, removing hop-by-hop headers."""
    headers = {}
    skip = {"host", "transfer-encoding", "content-length"}
    for key, value in request.headers.items():
        if key.lower() not in skip:
            headers[key] = value
    return headers


def _endpoint_for_path(path: str) -> str:
    """Determine the remapper endpoint label from the request path."""
    if "/chat/completions" in path:
        return "chat/completions"
    return "responses"


async def _stream_response(
    upstream_resp: "aiohttp.ClientResponse",
    downstream: web.StreamResponse,
) -> web.StreamResponse:
    """Pipe the upstream response body to the downstream stream."""
    downstream.content_type = upstream_resp.content_type
    downstream.status = upstream_resp.status
    try:
        chunk = await upstream_resp.content.readany()
        while chunk:
            await downstream.write(chunk)
            chunk = await upstream_resp.content.readany()
    except Exception:
        logger.exception("Error streaming upstream response")
    await downstream.write_eof()
    return downstream


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

async def health(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def handle_request(request: web.Request) -> web.StreamResponse:
    """Generic transparent proxy handler for all methods and paths."""
    path = request.path
    method = request.method
    forward_headers = _forward_headers(request)
    upstream = f"{request.app['upstream_url']}{path}"

    # Read request body (may be empty for GET/DELETE etc.)
    body = None
    if request.can_read_body:
        body = await request.read()

    namespace_map: dict = {}

    # Intercept transformable routes
    if method == "POST" and path in TRANSFORM_ROUTES and body:
        try:
            parsed = json.loads(body)
            parsed, namespace_map = await transform_tools_request(parsed)
            body = json.dumps(parsed).encode()
            # Update content-length since we modified the body
            forward_headers["content-length"] = str(len(body))
        except (json.JSONDecodeError, Exception):
            logger.warning("Failed to parse/transform request body for %s", path)

    # Store namespace_map keyed by request id so the response handler can
    # retrieve it (aiohttp request objects have a stable id for their lifetime).
    if namespace_map:
        _namespace_maps[id(request)] = namespace_map

    logger.info("%s %s -> %s", method, path, upstream)

    session: ClientSession = request.app["client_session"]
    try:
        async with session.request(
            method=method,
            url=upstream,
            headers=forward_headers,
            data=body,
            allow_redirects=False,
        ) as upstream_resp:
            is_sse = (
                upstream_resp.content_type
                and "text/event-stream" in upstream_resp.content_type
            )
            needs_response_transform = (
                path in TRANSFORM_ROUTES
                and namespace_map
                and not is_sse
                and upstream_resp.status == 200
            )

            if needs_response_transform:
                raw = await upstream_resp.read()
                # Clean up per-request namespace map
                _namespace_maps.pop(id(request), None)
                try:
                    resp_body = json.loads(raw)
                    endpoint = _endpoint_for_path(path)
                    resp_body = await transform_tools_response(
                        resp_body, namespace_map, endpoint=endpoint,
                    )
                    return web.Response(
                        body=json.dumps(resp_body).encode(),
                        content_type=upstream_resp.content_type,
                        status=upstream_resp.status,
                    )
                except (json.JSONDecodeError, Exception):
                    logger.warning(
                        "Failed to transform response body for %s", path,
                    )
                    return web.Response(
                        body=raw,
                        content_type=upstream_resp.content_type,
                        status=upstream_resp.status,
                    )

            # Streaming or non-transformable: pass through as-is.
            # For streaming responses we do NOT try to transform individual
            # chunks -- tool calls in streaming come as delta events and are
            # too complex to reliably rewrite on the fly.
            _namespace_maps.pop(id(request), None)
            downstream = web.StreamResponse()
            await downstream.prepare(request)
            return await _stream_response(upstream_resp, downstream)

    except Exception:
        _namespace_maps.pop(id(request), None)
        logger.exception("Upstream request failed for %s %s", method, path)
        return web.Response(status=502, text="Bad Gateway")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

async def on_startup(app: web.Application) -> None:
    app["client_session"] = ClientSession()
    logger.info("ClientSession created, upstream=%s", app["upstream_url"])


async def on_cleanup(app: web.Application) -> None:
    session: ClientSession = app["client_session"]
    await session.close()
    logger.info("ClientSession closed")


def create_app(config: Config | None = None) -> web.Application:
    """Build the aiohttp Application with proxy routes.

    Args:
        config: Optional Config instance. If None, defaults from env vars.
    """
    if config is None:
        config = Config()

    app = web.Application()
    app["upstream_url"] = config.UPSTREAM_URL
    app["config"] = config

    app.router.add_get("/health", health)
    # Catch-all for all methods and paths
    app.router.add_route("*", "/{path:.*}", handle_request)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


def main() -> None:
    """Entry point: configure logging and start the proxy."""
    config = Config()
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = create_app(config)
    logger.info("Starting proxy on port %d -> %s", config.LISTEN_PORT, config.UPSTREAM_URL)
    web.run_app(app, host="0.0.0.0", port=config.LISTEN_PORT, print=logger.info)


if __name__ == "__main__":
    main()
