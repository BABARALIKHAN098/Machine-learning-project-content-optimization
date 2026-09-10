"""One admitted operation from body receipt through completed worker serialization."""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from .errors import APIError
from .schemas import decode_request, request_frame, validate_response


class PredictionService:
    def __init__(self, predictor, settings):
        if (
            predictor.manifest.get("purpose") != "research"
            or predictor.manifest.get("production_ready") is not False
        ):
            raise ValueError("Unsupported application package")
        self.predictor = predictor
        limit = min(settings.maximum_batch_rows, predictor.config["maximum_batch_rows"])
        self.settings = replace(
            settings, maximum_batch_rows=limit, chunk_rows=min(settings.chunk_rows, limit)
        )
        self.ready = True
        self.gate = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-inference")

    def acquire(self):
        if not self.ready:
            raise APIError("service_not_ready")
        if not self.gate.acquire(blocking=False):
            raise APIError("prediction_capacity_exceeded")

    def process(self, body):
        started = time.perf_counter()
        frame, probabilities = request_frame(decode_request(body), self.predictor, self.settings)
        validated = time.perf_counter()
        try:
            result = self.predictor.predict(
                frame, include_probabilities=probabilities, chunk_rows=self.settings.chunk_rows
            )
            predicted = time.perf_counter()
            validate_response(result, frame, probabilities, self.predictor)
            serialized = bytearray()
            for part in json.JSONEncoder(allow_nan=False, ensure_ascii=True).iterencode(result):
                encoded = part.encode("utf-8")
                if len(serialized) + len(encoded) > self.settings.maximum_response_bytes:
                    raise APIError("response_limit_exceeded")
                serialized.extend(encoded)
            return bytes(serialized), {
                "row_count": len(frame),
                "include_probabilities": probabilities,
                "validation_seconds": validated - started,
                "prediction_seconds": predicted - validated,
                "serialization_seconds": time.perf_counter() - predicted,
            }
        except APIError:
            self.ready = False
            raise
        except Exception:  # noqa: BLE001 -- sanitize the public application boundary
            self.ready = False
            raise APIError("prediction_failed") from None

    async def run(self, body):
        try:
            future = self.executor.submit(self.process, body)
        except RuntimeError:
            self.gate.release()
            raise APIError("service_not_ready") from None
        future.add_done_callback(lambda completed: self.gate.release())
        wrapped = asyncio.wrap_future(future)
        # Retrieve detached failures too, so asyncio never logs a raw task traceback.
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(wrapped)

    async def close(self):
        self.ready = False
        await asyncio.to_thread(self.executor.shutdown, wait=True)
        await asyncio.to_thread(self.gate.acquire)
        self.gate.release()
        self.predictor = None
