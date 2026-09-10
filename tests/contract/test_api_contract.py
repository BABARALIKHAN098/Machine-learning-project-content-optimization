import copy
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import request_frame
from tests.api_helpers import TOKEN, fixture, headers


def client_for(predictor, settings):
    return TestClient(
        create_app(settings, lambda *a, **k: predictor), base_url="http://127.0.0.1:8000"
    )


def test_endpoints_docs_and_exact_prediction_parity(caplog):
    predictor, settings, document = fixture()
    with client_for(predictor, settings) as client:
        assert client.get("/health").json() == {"status": "healthy"}
        assert client.get("/ready").json()["production_ready"] is False
        assert client.get("/v1/model").status_code == 401
        assert client.get("/docs").status_code == 200
        assert "localStorage" not in client.get("/docs.js").text
        schema = client.get("/v1/schema", headers=headers()).json()
        assert schema["request"]["properties"]["records"]["items"]["additionalProperties"] is False
        openapi = client.get("/openapi.json").json()
        assert openapi["paths"]["/v1/predictions"]["post"]["security"]
        model = client.get("/v1/model", headers=headers()).json()
        assert model["recommended_model"] is None and "package_dir" not in model
        for probability in (False, True):
            document["include_probabilities"] = probability
            frame, _ = request_frame(document, predictor, settings)
            expected = predictor.predict(frame, include_probabilities=probability, chunk_rows=3)
            predictor._predictor.pipeline.calls.clear()
            response = client.post("/v1/predictions", headers=headers(), json=document)
            assert response.status_code == 200, response.text
            assert response.json() == expected
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-api-contract-version"] == "1.0"
            calls = predictor._predictor.pipeline.calls
            assert sum(c[0] == "predict" for c in calls) == 2
            assert sum(c[0] == "probability" for c in calls) == (2 if probability else 0)
    assert TOKEN not in caplog.text


@pytest.mark.parametrize(
    "change,status",
    [
        ({"Authorization": "wrong"}, 401),
        ({"Origin": "https://foreign.example"}, 403),
        ({"Host": "foreign.example"}, 400),
        ({"Content-Type": "text/plain"}, 415),
        ({"Content-Encoding": "gzip"}, 415),
    ],
)
def test_transport_rejections(change, status):
    predictor, settings, document = fixture()
    with client_for(predictor, settings) as client:
        response = client.post("/v1/predictions", headers={**headers(), **change}, json=document)
        assert response.status_code == status
        assert not predictor._predictor.pipeline.calls


def test_sanitized_errors_and_output_failure(caplog):
    predictor, settings, document = fixture()
    private = "seeded-private-identifier"
    document["records"][2]["content_id"] = private
    with client_for(predictor, settings) as client:
        bad = copy.deepcopy(document)
        bad["records"][2]["value"] = private
        result = client.post("/v1/predictions", headers=headers(), json=bad)
        assert result.status_code == 422 and private not in result.text
        assert not predictor._predictor.pipeline.calls
        assert client.get("/ready").status_code == 200
        predictor._predictor.pipeline.predict = lambda frame: [private] * len(frame)
        response = client.post("/v1/predictions", headers=headers(), json=document)
        assert response.status_code == 500 and private not in response.text
        assert "records" not in response.json()
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200
    assert private not in caplog.text


def test_response_cap_no_partial():
    predictor, settings, document = fixture()
    with client_for(predictor, replace(settings, maximum_response_bytes=10)) as client:
        response = client.post("/v1/predictions", headers=headers(), json=document)
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "response_limit_exceeded"


def test_whole_response_identity_validation():
    predictor, settings, document = fixture()
    original = predictor.predict

    def invalid(*args, **kwargs):
        result = original(*args, **kwargs)
        result["package_id"] = "forged"
        return result

    predictor.predict = invalid
    with client_for(predictor, settings) as client:
        assert client.post("/v1/predictions", headers=headers(), json=document).status_code == 500


def test_json_duplicate_and_missing_row_key():
    predictor, settings, document = fixture()
    with client_for(predictor, settings) as client:
        body = json.dumps(document).replace(
            '"purpose": "research"', '"purpose": "research", "purpose": "research"'
        )
        assert client.post("/v1/predictions", headers=headers(), content=body).status_code == 400
        document["records"][1].pop("kind")
        assert client.post("/v1/predictions", headers=headers(), json=document).status_code == 422
        assert not predictor._predictor.pipeline.calls
