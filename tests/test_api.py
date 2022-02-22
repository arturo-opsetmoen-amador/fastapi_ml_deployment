import pytest
from fastapi.testclient import TestClient
from main import app
import json


def test_api_locally_get_root():
    with TestClient(app) as client:
        r = client.get('/')
        assert r.status_code == 200, "Status code is not 200"
        assert r.json()["Welcome"] == "Welcome to the salary XGBoost predictor web service.", "Wrong json output"


def test_positive(positive_ex):
    with TestClient(app) as client:
        r = client.post("/predict", data=json.dumps(positive_ex),
                        headers={"Content-Type": "application/json"})
        assert r.json()["Salary"] == "1", "Unexpected output of the model"


def test_negative(negative_ex):
    with TestClient(app) as client:
        r = client.post("/predict", data=json.dumps(negative_ex),
                        headers={"Content-Type": "application/json"})
        assert r.json()["Salary"] == "0", "Unexpected output of the model"


def test_response(positive_ex):
    with TestClient(app) as client:
        r = client.post("/predict", data=json.dumps(positive_ex),
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 200
