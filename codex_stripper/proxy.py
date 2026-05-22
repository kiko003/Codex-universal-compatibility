"""Aiohttp-based reverse proxy for OpenAI Codex CLI tool transformation."""

import json
import logging
import os

from aiohttp import web, ClientSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("codex_stripper.proxy")

UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://localhost:4000")
PORT = int(os.environ.get("PROXY_PORT", "8080"))

# Routes that need tool namespace transformation
TRANSFORM_ROUTES = {"/v1/responses", "/v1/chat/completions"}


# ---------------------------------------------------------------------------
# Placeholder transformation hooks
# ---------------------------------------------------------------------------

async def transform_tools_request(body: dict) -> tuple[dict, dict]:
    """Transform outgoing request body for tool namespace flattening.

    Returns:
        (transformed_body, namespace_map) – namespace_map is forwarded to
        ``transform_tools_response`` so the response can be un-flattened.
    """
    return body, {}


async def transform_tools_response(body: dict, namespace_map: dict) -> dict:
    """Transform upstream response body, reversing namespace flattening."""
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
    upstream = f"{UPSTREAM_URL}{path}"

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
            # Update content-length if we modified the body
            forward_headers["content-length"] = str(len(body))
        except (json.JSONDecodeError, Exception):
            logger.warning("Failed to parse/transform request body for %s", path)

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

            # For transformable routes with non-streaming responses, we
            # may need to transform the response body too.
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
                try:
                    resp_body = json.loads(raw)
                    resp_body = await transform_tools_response(
                        resp_body, namespace_map
                    )
                    return web.Response(
                        body=json.dumps(resp_body).encode(),
                        content_type=upstream_resp.content_type,
                        status=upstream_resp.status,
                    )
                except (json.JSONDecodeError, Exception):
                    logger.warning("Failed to transform response body for %s", path)
                    return web.Response(
                        body=raw,
                        content_type=upstream_resp.content_type,
                        status=upstream_resp.status,
                    )

            # Default: stream the response through
            downstream = web.StreamResponse()
            await downstream.prepare(request)
            return await _stream_response(upstream_resp, downstream)

    except Exception:
        logger.exception("Upstream request failed for %s %s", method, path)
        return web.Response(status=502, text="Bad Gateway")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

async def on_startup(app: web.Application) -> None:
    app["client_session"] = ClientSession()
    logger.info("ClientSession created, upstream=%s", UPSTREAM_URL)


async def on_cleanup(app: web.Application) -> None:
    session: ClientSession = app["client_session"]
    await session.close()
    logger.info("ClientSession closed")


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    # Catch-all for all methods and paths
    app.router.add_route("*", "/{path:.*}", handle_request)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    app = create_app()
    logger.info("Starting proxy on port %d -> %s", PORT, UPSTREAM_URL)
    web.run_app(app, host="0.0.0.0", port=PORT, print=logger.info)


if __name__ == "__main__":
    main()
