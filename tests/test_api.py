import pytest
from fastapi.testclient import TestClient
from main import app
import json
import logging


def test_api_get_root():
    """
    Tests the GET method for the implemented API.
    Returns: None
    -------

    """
    with TestClient(app) as client:
        try:
            r = client.get('/')
            assert r.status_code == 200
            logging.info(
                "Testing GET method API: Status code 200 delivered: SUCCESS"
            )
        except AssertionError as err:
            logging.error(
                "Testing GET method API: Status code 200 not delivered: ERROR"
            )
            raise err


def test_api_get_message():
    """
    Tests the GET method for the implemented API.
    Returns: None
    -------

    """
    with TestClient(app) as client:
        try:
            r = client.get('/')
            assert r.json()["Welcome"] == "Welcome to the salary XGBoost predictor web service."
            logging.info(
                "Testing GET method Welcome message: The message delivered is correct: SUCCESS"
            )
        except AssertionError as err:
            logging.error(
                "Testing GET method Welcome message: The message delivered is not correct: ERROR"
            )
            raise err


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
