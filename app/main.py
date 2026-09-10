"""Local research API. Importing this module never loads a model."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from starlette.exceptions import HTTPException

from machine_learning_project.inference.packaged_predictor import PackagedPredictor

from .config import ConfigurationError, load_settings
from .errors import ERRORS, APIError, error_response
from .middleware import TransportMiddleware
from .routes import router
from .schemas import request_schema
from .service import PredictionService


def create_app(settings=None, predictor_loader=None):
    @asynccontextmanager
    async def lifespan(application):
        try:
            effective = settings or load_settings()
            predictor = (predictor_loader or PackagedPredictor.load)(
                effective.package_dir,
                purpose=effective.purpose,
                expected_manifest_sha256=effective.expected_manifest_sha256,
            )
            service = PredictionService(predictor, effective)
        except Exception:  # noqa: BLE001 -- sanitize the public application boundary
            raise ConfigurationError("Research API startup verification failed") from None
        application.state.settings = service.settings
        application.state.service = service
        document = get_openapi(
            title="Content Trend Research API", version="1.0", routes=application.routes
        )
        document["components"] = {
            "securitySchemes": {"ResearchToken": {"type": "http", "scheme": "bearer"}}
        }
        for path in ("/v1/model", "/v1/schema", "/v1/predictions"):
            for operation in document["paths"][path].values():
                operation["security"] = [{"ResearchToken": []}]
                operation["responses"].update(
                    {str(status): {"description": message} for status, message in ERRORS.values()}
                )
        operation = document["paths"]["/v1/predictions"]["post"]
        operation["requestBody"] = {
            "required": True,
            "content": {
                "application/json": {"schema": request_schema(predictor.schema, service.settings)}
            },
        }
        operation["description"] = (
            "Research-only inference v2 envelope. Input order preserved. Probability fields are null by default; opt-in scores are uncalibrated and non-causal."
        )
        application.state.openapi = document
        try:
            yield
        finally:
            await service.close()

    application = FastAPI(
        title="Content Trend Research API",
        version="1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        debug=False,
    )
    application.include_router(router)
    application.add_middleware(TransportMiddleware)

    @application.exception_handler(APIError)
    async def api_error(request, error):
        return error_response(error.code, request.state.request_id)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return error_response("invalid_request", request.state.request_id)

    @application.exception_handler(HTTPException)
    async def http_error(request, error):
        code = {404: "not_found", 405: "method_not_allowed"}.get(
            error.status_code, "invalid_request"
        )
        return error_response(code, request.state.request_id)

    return application


app = create_app()
