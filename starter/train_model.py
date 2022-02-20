"""
Add Doc string # TODO: Add doc string
"""
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import logging
from statistics import mean
import pandas as pd
from typing import *
from pathlib import Path
from ml.data import process_data
from ml.model import get_best_params, compute_model_metrics, inference, train_xgb
import pickle
from datetime import datetime
# TODO: Check logger
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
num_folds = 10
kfolds = KFold(n_splits=num_folds, random_state=None, shuffle=False)
X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

time_start = datetime.now()

best_params = get_best_params(X_train, y_train, kfolds)
xgb_model = train_xgb(X_train, y_train, best_params)
xgb_predictions = inference(xgb_model, X_test)
xgb_precision, xgb_recall, xgb_fbeta = compute_model_metrics(y_test, xgb_predictions)
print("xgb_Precisions: ", xgb_precision, "\n",
          "xgb_Recalls: ", xgb_recall, "\n",
          "xgb_FBetas: ", xgb_fbeta, "\n",
      )

time_end = datetime.now()
print("Running time: ", time_end-time_start)


# Proces the test data with the process_data function.

# Train and save a model.
