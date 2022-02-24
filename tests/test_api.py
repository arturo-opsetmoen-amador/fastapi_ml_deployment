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
    Tests the GET Welcome message for the implemented API.
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


def test_positive(positive_ex) -> None:
    """
    Tests the post/predict API method for our web app. Example taken from the dataset to ensure a positive
    prediction.
    Parameters
    ----------
    positive_ex: Fixture positive_ex as defined in conftest.py with session scope.

    Returns: None
    -------

    """
    with TestClient(app) as client:
        try:
            r = client.post("/predict", data=json.dumps(positive_ex), headers={"Content-Type": "application/json"})
            assert r.json()["Salary"] == "1"
            logging.info(
                "Testing POST/predict method: The prediction is correct: SUCCESS"
            )
        except AssertionError as err:
            logging.error(
                "Testing POST/predict method: The prediction is not correct: ERROR"
            )
            raise err




def test_negative(negative_ex) -> None:
    """
    Tests the post/predict API method for our web app. Example taken from the dataset to ensure a negative
    prediction.
    Parameters
    ----------
    negative_ex: Fixture positive_ex as defined in conftest.py with session scope.

    Returns: None
    -------

    """
    with TestClient(app) as client:
        try:
            r = client.post("/predict", data=json.dumps(negative_ex), headers={"Content-Type": "application/json"})
            assert r.json()["Salary"] == "0"
            logging.info(
                "Testing POST/predict method: The prediction is correct: SUCCESS"
            )
        except AssertionError as err:
            logging.error(
                "Testing POST/predict method: The prediction is not correct: ERROR"
            )
            raise err


def test_post_code(positive_ex):
    """
    Tests the post/predict API method for our web app. This tests checks the response code (200) from the API.
    Parameters
    ----------
    positive_ex: Fixture positive_ex as defined in conftest.py with session scope. Tests can also run with neg. fixture.

    Returns: None
    -------

    """
    with TestClient(app) as client:
        try:
            r = client.post("/predict", data=json.dumps(positive_ex), headers={"Content-Type": "application/json"})
            assert r.status_code == 200
            logging.info(
                "Testing POST/predict returned code: The code is correct: SUCCESS"
            )
        except AssertionError as err:
            logging.error(
                "Testing POST/predict returned code: The code is not correct: SUCCESS"
            )
            raise err

