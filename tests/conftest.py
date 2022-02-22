import pytest
from typing import *
import logging
import pandas as pd
from pandas import errors as pde
from ml_training import train_model as tm


@pytest.fixture(name='input_dataframe', scope='session')
def input_dataframe_() -> pd.DataFrame:
    """Use fixtures to define the dataframe"""
    try:
        test_data = tm.read_data('data/census_clean_v1.csv')
        logging.info("Fixture: Test data imported successfully: SUCCESS")
    except pde.EmptyDataError as err:
        logging.error("Fixture: Testing import_data: Your data file is empty")
        raise err
    except FileNotFoundError as err:
        logging.error("Fixture: Testing import_data: The file wasn't found")
        raise err
    return test_data


@pytest.fixture(name='train_test_split', scope='session')
def train_test_split_(input_dataframe) -> Tuple[List[Any], List[Any]]:
    """

    Parameters
    ----------
    input_dataframe

    Returns
    -------

    """
    try:
        train, test = tm.split_data(input_dataframe)
        assert len(train) > 0
        assert len(test) > 0
        logging.info(
            "Fixture: Test data_split: SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Fixture: Test data_split does not produce the expected lists: ERROR"
        )
        raise err

    return train, test


@pytest.fixture(name='train_test_gen', scope='session')
def train_test_gen_(train_test_split):
    """

    Parameters
    ----------
    train_test_split

    Returns
    -------

    """
    try:
        train, test = train_test_split
        X_train, y_train, encoder_in, lb = tm.train_gen(train)
        X_test, y_test, encoder_test, lb_ = tm.test_gen(test, encoder_in, lb)

        assert len(X_train) > 0
        assert len(y_train) > 0
        assert len(X_test) > 0
        assert len(y_test) > 0
        logging.info(
            "Fixture: Generate train, test split with tm.train_gen: SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Fixture: Test train_gen, test_gen does not produce the expected lists: ERROR"
        )
    return X_train, y_train, X_test, y_test, encoder_in, lb


@pytest.fixture(name='test_xgb_pipe', scope='session')
def test_xgb_pipe_(train_test_gen):
    """

    Parameters
    ----------
    train_test_gen

    Returns
    -------

    """
    X_train, y_train, X_test, y_test, encoder, lb = train_test_gen
    try:
        xgb_model = tm.train_xgb_pipe(X_train, y_train, X_test, y_test)
        assert len(xgb_model.predict(X_test)) > 0
        logging.info(
            "Testing train_xgb_pipe: The model delivers predictions: SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing train_xgb_pipe: The model doesn't deliver predictions: ERROR"
        )
        raise err

    return xgb_model, encoder, lb


@pytest.fixture(name="negative_ex", scope='session')
def negative_ex_():
    """

    Returns
    -------

    """
    test_input = {
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    return test_input


@pytest.fixture(name="positive_ex", scope='session')
def positive_ex_():
    """

    Returns
    -------

    """
    test_input = {
        "age": 40,
        "workclass": "Private",
        "education": "Doctorate",
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    return test_input
