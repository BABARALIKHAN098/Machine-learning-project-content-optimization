"""ASGI boundary with allowlisted logs and no raw URLs or credentials."""

import json
import logging
import secrets
import time
import uuid

from .errors import APIError, error_response

ROUTES = {
    "/health",
    "/ready",
    "/docs",
    "/docs.js",
    "/openapi.json",
    "/v1/model",
    "/v1/schema",
    "/v1/predictions",
}
PROTECTED = {"/v1/model", "/v1/schema", "/v1/predictions"}
logger = logging.getLogger("research_api")


class TransportMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        status, sent = 500, False
        code = None

        async def safe_send(message):
            nonlocal status, sent
            if message["type"] == "http.response.start":
                sent = True
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"cache-control", b"no-store"),
                    (b"x-request-id", request_id.encode()),
                    (b"x-api-contract-version", b"1.0"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (
                        b"content-security-policy",
                        b"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
                    ),
                ]
            await send(message)

        try:
            settings = getattr(scope["app"].state, "settings", None)
            if settings is None:
                raise APIError("service_not_ready")
            pairs = scope.get("headers", [])
            for key in (
                b"host",
                b"authorization",
                b"origin",
                b"content-type",
                b"content-length",
                b"content-encoding",
            ):
                if sum(k == key for k, v in pairs) > 1:
                    raise APIError("invalid_request")
            headers = dict(pairs)
            host = headers.get(b"host", b"").decode("latin-1")
            if host not in {
                "127.0.0.1",
                f"127.0.0.1:{settings.port}",
                "localhost",
                f"localhost:{settings.port}",
            }:
                raise APIError("invalid_host")
            if scope["path"] in PROTECTED:
                expected = b"Bearer " + settings.token.encode("ascii")
                if not secrets.compare_digest(headers.get(b"authorization", b""), expected):
                    raise APIError("unauthorized")
                if (
                    b"origin" in headers
                    and headers[b"origin"].decode("latin-1") != f"http://{host}"
                ):
                    raise APIError("invalid_origin")
            if scope["path"] == "/v1/predictions" and scope["method"] == "POST":
                content_type = (
                    headers.get(b"content-type", b"").decode("latin-1").lower().replace(" ", "")
                )
                if (
                    content_type not in {"application/json", "application/json;charset=utf-8"}
                    or b"content-encoding" in headers
                ):
                    raise APIError("unsupported_media_type")
                length = headers.get(b"content-length")
                if length is not None:
                    if not length.isdigit():
                        raise APIError("invalid_request")
                    if len(length) > 12 or int(length) > settings.maximum_request_bytes:
                        raise APIError("request_too_large")
            await self.app(scope, receive, safe_send)
        except APIError as error:
            code = error.code
            if not sent:
                await error_response(code, request_id)(scope, receive, safe_send)
        except Exception:  # noqa: BLE001 -- sanitize the public application boundary
            code = "prediction_failed"
            if not sent:
                await error_response(code, request_id)(scope, receive, safe_send)
        finally:
            event = {
                "request_id": request_id,
                "route": scope["path"] if scope["path"] in ROUTES else "unmatched",
                "status": status,
                "duration_seconds": round(time.perf_counter() - started, 6),
            }
            if code:
                event["error_code"] = code
            event.update(scope["state"].get("metrics", {}))
            logger.info(json.dumps(event, allow_nan=False))
