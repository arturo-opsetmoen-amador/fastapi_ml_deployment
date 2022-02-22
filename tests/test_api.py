import pytest
from fastapi.testclient import TestClient
from ..fastAPI_main import app
import json

def test_positive_example(data_positive_API):#, compare_output):#, app):
    with TestClient(app) as client:
            r = client.post("/predict", data=json.dumps(data_positive_API),
                            headers={"Content-Type": "application/json"})
            assert r.json()["predictions"] == "1", "Unexpected output of the model"

def test_negative_example(data_negative_API):#, compare_output):#, app):
    with TestClient(app) as client:
            r = client.post("/predict", data=json.dumps(data_negative_API),
                            headers={"Content-Type": "application/json"})
            assert r.json()["predictions"] == "0", "Unexpected output of the model"

def test_response_code(data_positive_API):
    with TestClient(app) as client:
            r = client.post("/predict", data=json.dumps(data_positive_API),
                            headers={"Content-Type": "application/json"})
            assert r.status_code == 200