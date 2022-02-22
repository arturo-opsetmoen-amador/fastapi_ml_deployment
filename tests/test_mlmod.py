import pandas as pd
import pytest
import numpy as np
from typing import *
import logging
from eda import data_cleaning as dc
from ml_training import train_model as tm

print(__name__)


def test_column_names_raw(input_dataframe: pd.DataFrame):
    """
    Tests whether the column names and column order are as expected
    """
    expected_columns = ["age", "workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "sex",
                        "hours-per-week", "native-country",
                        "salary"]
    try:
        assert list(expected_columns) == list(input_dataframe.columns.values)
        logging.info(
            "Testing read_data: The census data was read in correctly: SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing read_data: The file doesn't appear to have rows or columns"
        )
        raise err


def test_split_data(input_dataframe: pd.DataFrame):
    """

    Parameters
    ----------
    test_dataframe

    Returns
    -------

    """
    train, test = tm.split_data(input_dataframe)
    try:
        assert train.shape[0] > 0
        assert test.shape[0] > 0
        assert train.shape[1] > 0
        assert test.shape[1] > 0
        logging.info(
         "Testing split_data: The census data was split correctly: SUCCESS"
            )
    except AssertionError as err:
        logging.error(
            "Testing split_data: The split seems to be empty: ERROR"
        )
        raise err


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


def test_sliced_metrics(test_xgb_pipe, input_dataframe):
    """

    Parameters
    ----------
    test_xgb_pipe

    Returns
    -------

    """
    try:
        xgb_model, encoder, lb = test_xgb_pipe
        data = input_dataframe
        feature = 'race'
        assert tm.sliced_metrics(data, feature, xgb_model, encoder, lb)
        logging.info(
            "Testing sliced_metrics: Predictions for the feature race are delivered with "
            "precission greater than zero: SUCCESS"
        )
    except AssertionError as err:
        logging.error(
            "Testing sliced_metrics: Predictions for the feature race were not delivered: ERROR"
        )



