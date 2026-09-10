"""Public error vocabulary; never interpolate exceptions or submitted values."""

from starlette.responses import JSONResponse

ERRORS = {
    "invalid_json": (400, "Request must contain valid strict UTF-8 JSON."),
    "invalid_request": (422, "Request does not match the inference contract."),
    "unauthorized": (401, "A valid bearer token is required."),
    "request_body_timeout": (408, "Request body receipt timed out."),
    "request_too_large": (413, "Request exceeds the configured byte limit."),
    "unsupported_media_type": (415, "Only uncompressed UTF-8 JSON is supported."),
    "prediction_capacity_exceeded": (429, "Prediction capacity is busy. Retry later."),
    "prediction_failed": (500, "Prediction could not be completed."),
    "response_limit_exceeded": (500, "Response exceeds the configured byte limit."),
    "service_not_ready": (503, "Research prediction service is not ready."),
    "invalid_host": (400, "Request host is not supported."),
    "invalid_origin": (403, "Request origin is not supported."),
    "not_found": (404, "Endpoint not found."),
    "method_not_allowed": (405, "Method not allowed."),
}


class APIError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def error_response(code, request_id):
    status, message = ERRORS[code]
    headers = {}
    if status == 401:
        headers["WWW-Authenticate"] = "Bearer"
    if status == 429:
        headers["Retry-After"] = "1"
    return JSONResponse(
        {"error": {"code": code, "message": message, "request_id": request_id}},
        status_code=status,
        headers=headers,
    )
