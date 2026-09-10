import asyncio
import json
import threading
from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigurationError
from app.errors import APIError
from app.main import create_app
from tests.api_helpers import fixture, headers


def scope(path="/v1/predictions", method="POST"):
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "server": ("127.0.0.1", 8000),
        "client": ("127.0.0.1", 4444),
        "headers": [
            (b"host", b"127.0.0.1:8000"),
            *[(k.lower().encode(), v.encode()) for k, v in headers().items()],
        ],
    }


async def call(application, messages, path="/v1/predictions", method="POST"):
    output = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        output.append(message)

    await application(scope(path, method), receive, send)
    return output


def test_lifecycle_no_import_load_and_startup_failures():
    predictor, settings, _ = fixture()
    loader = Mock(return_value=predictor)
    application = create_app(settings, loader)
    loader.assert_not_called()
    with TestClient(application, base_url="http://127.0.0.1:8000") as client:
        assert client.get("/health").status_code == 200
        loader.assert_called_once_with(
            "synthetic", purpose="research", expected_manifest_sha256="1" * 64
        )
        assert not predictor._predictor.pipeline.calls
    assert application.state.service.ready is False
    for error in (ValueError("private-path"), FileNotFoundError("private-path")):
        with (
            pytest.raises(ConfigurationError, match="startup verification failed") as raised,
            TestClient(create_app(settings, Mock(side_effect=error))),
        ):
            pass
        assert "private-path" not in str(raised.value)


def test_streamed_limit_without_content_length_and_disconnect():
    async def scenario():
        predictor, settings, _ = fixture()
        application = create_app(
            replace(settings, maximum_request_bytes=10), lambda *a, **k: predictor
        )
        async with application.router.lifespan_context(application):
            output = await call(
                application,
                [
                    {"type": "http.request", "body": b"123456", "more_body": True},
                    {"type": "http.request", "body": b"789012", "more_body": False},
                ],
            )
            assert output[0]["status"] == 413
            assert not application.state.service.gate.locked()
            output = await call(application, [{"type": "http.disconnect"}])
            assert not application.state.service.gate.locked()
            assert not predictor._predictor.pipeline.calls

    asyncio.run(scenario())


def test_body_timeout_releases_admission():
    async def scenario():
        predictor, settings, _ = fixture()
        application = create_app(
            replace(settings, body_timeout_seconds=1), lambda *a, **k: predictor
        )
        async with application.router.lifespan_context(application):
            output = []

            async def receive():
                await asyncio.sleep(2)
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                output.append(message)

            await application(scope(), receive, send)
            assert output[0]["status"] == 408
            assert not application.state.service.gate.locked()

    asyncio.run(scenario())


def test_cancelled_http_retains_worker_lease_health_and_shutdown():
    async def scenario():
        predictor, settings, document = fixture()
        entered, release = threading.Event(), threading.Event()
        original = predictor.predict

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)

        predictor.predict = blocked
        application = create_app(settings, lambda *a, **k: predictor)
        async with application.router.lifespan_context(application):
            service = application.state.service
            request = {
                "type": "http.request",
                "body": json.dumps(document).encode(),
                "more_body": False,
            }
            pending = asyncio.create_task(call(application, [request]))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                live = await call(application, [], "/health", "GET")
                assert live[0]["status"] == 200
                busy = await call(application, [request])
                assert busy[0]["status"] == 429
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert service.gate.locked()
                with pytest.raises(APIError, match="capacity"):
                    service.acquire()
                closing = asyncio.create_task(service.close())
                await asyncio.sleep(0.02)
                assert not closing.done() and service.ready is False
            finally:
                release.set()
            await asyncio.wait_for(closing, 5)
            assert not service.gate.locked()

    asyncio.run(scenario())
