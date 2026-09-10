"""Transport routes: body admission precedes all CPU work."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from machine_learning_project.inference.contracts import OUTPUT_SCHEMA

from .errors import APIError
from .schemas import request_schema, synthetic_document

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.get("/ready")
async def ready(request: Request):
    if not request.app.state.service.ready:
        raise APIError("service_not_ready")
    return {"status": "ready", "purpose": "research", "production_ready": False}


@router.get("/v1/model")
async def model(request: Request):
    service = request.app.state.service
    if not service.ready:
        raise APIError("service_not_ready")
    predictor = service.predictor
    manifest = predictor.manifest
    return {
        **{
            k: manifest[k]
            for k in (
                "package_id",
                "model_version",
                "family",
                "purpose",
                "production_ready",
                "recommended_model",
                "development_reference",
            )
        },
        "package_manifest_sha256": predictor.manifest_sha256,
        "research_only": True,
        "limitations": [
            "Research use only; human review required.",
            "Probabilities are uncalibrated and non-causal.",
            "Historical holdout exposure, cutoff review and production acceptance remain unresolved.",
        ],
        "limits": public_limits(service.settings),
    }


def public_limits(settings):
    return {
        key: getattr(settings, key)
        for key in (
            "maximum_batch_rows",
            "chunk_rows",
            "maximum_request_bytes",
            "maximum_response_bytes",
            "maximum_content_id_bytes",
            "maximum_category_bytes",
            "maximum_active_predictions",
            "body_timeout_seconds",
        )
    }


@router.get("/v1/schema")
async def schema(request: Request):
    service = request.app.state.service
    if not service.ready:
        raise APIError("service_not_ready")
    return {
        "input": service.predictor.schema,
        "output": OUTPUT_SCHEMA,
        "request": request_schema(service.predictor.schema, service.settings),
        "limits": public_limits(service.settings),
        "example": synthetic_document(service.predictor.schema, 2),
    }


@router.post("/v1/predictions")
async def predictions(request: Request):
    service = request.app.state.service
    service.acquire()
    transferred = False
    try:
        body = bytearray()
        try:
            async with asyncio.timeout(service.settings.body_timeout_seconds):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > service.settings.maximum_request_bytes:
                        raise APIError("request_too_large")
                    body.extend(chunk)
        except TimeoutError:
            raise APIError("request_body_timeout") from None
        # The task starts synchronously on this loop before its first await.
        transferred = True
        payload, metrics = await service.run(bytes(body))
        request.scope["state"]["metrics"] = metrics
        return Response(payload, media_type="application/json")
    finally:
        if not transferred:
            service.gate.release()


@router.get("/docs", include_in_schema=False)
async def docs():
    return HTMLResponse((Path(__file__).parent / "static/docs.html").read_text(encoding="utf-8"))


@router.get("/docs.js", include_in_schema=False)
async def docs_script():
    return Response(
        (Path(__file__).parent / "static/docs.js").read_bytes(), media_type="application/javascript"
    )


@router.get("/openapi.json", include_in_schema=False)
async def openapi(request: Request):
    return JSONResponse(request.app.state.openapi)
