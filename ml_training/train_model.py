"""
Add Doc string # TODO: Add doc string
"""
import joblib
import xgboost
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import logging
from statistics import mean
from xgboost import XGBClassifier
import pandas as pd
from typing import *
from pathlib import Path
from ml.data import process_data
from ml.model import get_best_params, compute_model_metrics, inference, train_xgb
import sys
sys.path.insert(1, '../data')


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


FORMAT = '%(asctime)-15s %(message)s'
formatter = logging.Formatter(FORMAT)


def logger(name: str, log_file: str, level:logging.INFO=logging.INFO) -> logging.Logger:
    """

    Parameters
    ----------
    name
    log_file
    level

    Returns
    -------

    """
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger_ = logging.getLogger(name)
    logger_.setLevel(level)
    logger_.addHandler(handler)

    return logger_


train_log = logger('train_log', 'train_log.txt')
slices_log = logger('slices_log', "slices_log.txt")

# TODO: Check logger


def sliced_metrics(data_frame: pd.DataFrame, feature: str, model: XGBClassifier, encoding: OneHotEncoder,
                   labels: LabelBinarizer) -> None:
    """
    Parameters
    ----------
    data_frame
    feature
    model
    encoding
    labels
    Returns
    -------
    """
    for classes in data_frame[feature].unique():
        data_frame_temp = data_frame[data_frame[feature] == classes]
        x_set, y_set, _, _ = process_data(data_frame_temp, categorical_features=cat_features, label="salary",
                                          training=False,
                                          encoder=encoding, lb=labels)

        predicted_values = inference(model, x_set)
        precision, recall, fbeta = compute_model_metrics(y_set, predicted_values)
        slices_log.info(f"Feature: {feature}")
        slices_log.info(f"Class: {classes}")
        slices_log.info(f"{feature} precision: {precision}")
        slices_log.info(f"{feature} recall: {recall}")
        slices_log.info(f"{feature} fbeta: {fbeta}\n")

    return None


def read_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """

    Parameters
    ----------
    data_path

    Returns
    -------

    """
    # data = pd.read_csv("../data/census_clean_v1.csv")
    data_frame = pd.read_csv(data_path)
    return data_frame


def split_data(data_frame):
    train_, test_ = train_test_split(data_frame, test_size=0.2, random_state=311, stratify=data_frame['salary'])
    return train_, test_


def train_gen(train_set):
    x_train_, y_train_, encoder_, lb_ = process_data(train_set, categorical_features=cat_features, label="salary", training=True)
    return x_train_, y_train_, encoder_, lb_


def test_gen(test_set, encoder, lb):
    x_test, y_test_, encoder_, lb_ = process_data(test_set, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    return x_test, y_test_, encoder_, lb_


def train_xgb_pipe(X_train, y_train, X_test, y_test, num_folds=10, search=False) -> xgboost.XGBClassifier:
    """

    Parameters
    ----------
    data_path
    search
    num_folds
    data_frame

    Returns
    -------

    """
    k_folds = StratifiedKFold(n_splits=num_folds, random_state=None, shuffle=False)
    if search:
        best_params = get_best_params(X_train, y_train, k_folds)
    else:
        best_params = joblib.load('../model/best_params_20220221_103639.pkl')
    xgb_model = train_xgb(X_train, y_train, best_params)
    xgb_predictions = inference(xgb_model, X_test)
    xgb_precision, xgb_recall, xgb_fbeta = compute_model_metrics(y_test, xgb_predictions)

    train_log.info(f"Test xgb_Precision: {xgb_precision}")
    train_log.info(f"Test xgb_Recall: {xgb_recall}")
    train_log.info(f"Test xgb_FBeta: {xgb_fbeta}")
    train_log.info("\n")
    return xgb_model


if __name__=="__main__":
    data = read_data('../data/census_clean_v1.csv')
    train, test = split_data(data)
    X_train, y_train, encoder, lb = train_gen(train)
    X_test, y_test, encoder_, lb_ = test_gen(test, encoder, lb)
    xgb_model = train_xgb_pipe(X_train, y_train, X_test, y_test)
    for feature in cat_features:
        sliced_metrics(data, feature, xgb_model, encoder, lb)
