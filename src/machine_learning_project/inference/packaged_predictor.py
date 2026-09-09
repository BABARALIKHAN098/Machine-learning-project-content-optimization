"""Verified local package inference; never refits or changes the frozen estimator."""

import copy

from threadpoolctl import threadpool_limits

from ..data.ingestion import sha256_file
from ..models.package_artifacts import (
    load_package_manifest,
    local_path,
    read_json,
    validate_model,
)
from ..utils.exceptions import DataValidationError
from .contracts import validate_predictions, validate_request
from .predictor import Predictor


class PackagedPredictor:
    @classmethod
    def load(cls, package_dir, *, purpose, expected_manifest_sha256=None):
        if purpose != "research":
            raise DataValidationError("Only explicit research inference is supported")
        root = local_path(package_dir)
        manifest = load_package_manifest(root, expected_manifest_sha256)
        # All bytes/schema/runtime checked before the trusted legacy bundle deserializer.
        predictor = Predictor.load(root / "model.joblib")
        schema = read_json(root / "input_schema.json")
        validate_model(predictor, schema, read_json(root / "model_metadata.json"))
        instance = cls()
        instance.root, instance.manifest = root, manifest
        instance.manifest_sha256 = sha256_file(root / "package_manifest.json")
        instance._predictor, instance.schema = predictor, schema
        instance.config = read_json(root / "inference_config.json")
        return instance

    def predict(self, frame, *, include_probabilities=False, chunk_rows=None):
        if type(include_probabilities) is not bool:
            raise DataValidationError("Probability mode must be boolean")
        config = copy.deepcopy(self.config)
        if chunk_rows is not None:
            if type(chunk_rows) is not int or not 1 <= chunk_rows <= config["maximum_batch_rows"]:
                raise DataValidationError("Invalid chunk size")
            config["chunk_rows"] = chunk_rows
        request = validate_request(frame, self.schema, config)
        pipeline = self._predictor.pipeline
        if include_probabilities and not hasattr(pipeline, "predict_proba"):
            raise DataValidationError("Requested probability capability is unavailable")
        records = []
        for start in range(0, len(request), config["chunk_rows"]):
            chunk = request.iloc[start : start + config["chunk_rows"]]
            features = chunk[self.schema["features"]]
            try:
                with threadpool_limits(limits=config["thread_limit"]):
                    predictions = pipeline.predict(features)
                    probabilities = (
                        pipeline.predict_proba(features) if include_probabilities else None
                    )
            except Exception:  # noqa: BLE001 -- estimator messages may expose private input values
                # Estimator errors can contain input values; the operational error is aggregate only.
                raise DataValidationError(
                    "model_inference: frozen pipeline could not score request"
                ) from None
            if include_probabilities and probabilities is None:
                raise DataValidationError("model_output: requested probabilities are missing")
            records.extend(
                validate_predictions(
                    chunk["content_id"].tolist(),
                    predictions,
                    probabilities,
                    list(map(str, pipeline.classes_)),
                    self.schema,
                    self._predictor.metadata,
                    config["probability_tolerance"],
                )
            )
        return {
            "inference_contract_version": "2.0",
            "package_schema_version": "1.0",
            "package_id": self.manifest["package_id"],
            "package_manifest_sha256": self.manifest_sha256,
            "model_version": self.manifest["model_version"],
            "purpose": "research",
            "production_ready": False,
            "row_count": len(records),
            "include_probabilities": include_probabilities,
            "probability_interpretation": "uncalibrated_noncausal"
            if include_probabilities
            else "not_requested",
            "records": records,
        }

    def rank_review_queue(self, frame, *, include_probabilities=True):
        if include_probabilities is not True:
            raise DataValidationError("Review ranking requires explicit probability mode")
        response = self.predict(frame, include_probabilities=True)
        response["records"] = sorted(response["records"], key=lambda r: -r["probability_down"])
        response["order"] = "descending_probability_down_stable_input_position_ties"
        return response
