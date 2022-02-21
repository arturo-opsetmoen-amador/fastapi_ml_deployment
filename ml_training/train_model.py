"""
Add Doc string # TODO: Add doc string
"""
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
import pickle
from datetime import datetime


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

        predicted_values = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, predicted_values)


formatter = logging.Formatter('%(asctime)-15s %(message)s')

data = pd.read_csv("../data/census_clean_v1.csv")

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

# KFold Cross validation
train, test = train_test_split(data, test_size=0.2, random_state=311, stratify=data['salary'])


X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

time_start = datetime.now()

num_folds = 10
kfolds = StratifiedKFold(n_splits=num_folds, random_state=None, shuffle=False)

best_params = get_best_params(X_train, y_train, kfolds)
xgb_model = train_xgb(X_train, y_train, best_params)
xgb_predictions = inference(xgb_model, X_test)
xgb_precision, xgb_recall, xgb_fbeta = compute_model_metrics(y_test, xgb_predictions)
print("xgb_Precisions: ", xgb_precision, "\n",
      "xgb_Recalls: ", xgb_recall, "\n",
      "xgb_FBetas: ", xgb_fbeta, "\n",
      )

time_end = datetime.now()
print("Running time: ", time_end - time_start)

# Proces the test data with the process_data function.

# Train and save a model.
