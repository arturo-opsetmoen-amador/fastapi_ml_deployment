"""
Add doc string # TODO: Add doc string
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
import logging
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Integer, Real
import xgboost as xgb
from datetime import datetime
from typing import *
from pathlib import Path
import joblib

FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'
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

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


params_log = logger('params_log', 'params_log.txt')


def get_best_params(X_train, y_train, cv,
                    filepath=f'../model/best_params_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'):
    """
    Finds the best parameters in the search space.
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Params dictionary
    """
    overdone_control = DeltaYStopper(delta=0.0001)
    time_limit_control = DeadlineStopper(total_time=60 * 45)
    callbacks = [overdone_control, time_limit_control
                 ]
    xgboost = xgb.XGBClassifier(n_jobs=-1,
                                objective='binary:logistic',
                                use_label_encoder=False,
                                eval_metric='logloss',
                                tree_method='gpu_hist')
    search_spaces = {
        'learning_rate': Real(0.01, 1.0, 'log-uniform'),
        'min_child_weight': Integer(0, 10),
        'max_depth': Integer(0, 100),
        'max_delta_step': Integer(0, 20),
        'subsample': Real(0.01, 1.0, 'uniform'),
        'colsample_bytree': Real(0.01, 1.0, 'log-uniform'),
        'colsample_bylevel': Real(0.01, 1.0, 'log-uniform'),
        'reg_lambda': Real(1e-9, 1000, 'log-uniform'),
        'reg_alpha': Real(1e-9, 1.0, 'log-uniform'),
        'gamma': Real(1e-9, 0.5, 'log-uniform'),
        'n_estimators': Integer(400, 1000),
        'scale_pos_weight': Real(1e-6, 500, 'log-uniform')
        }

    classifier = BayesSearchCV(
        xgboost,
        search_spaces,
        n_iter=4096,
        cv=cv
    )

    classifier.fit(X_train, y_train, callback=callbacks)

    best_params = classifier.best_params_
    params_log.info('Best parameters: ')
    params_log.info(f"{best_params}")
    joblib.dump(best_params, filepath)

    return best_params


def train_xgb(X_train, y_train, best_params, filepath: Union[
    str, Path] = f'../model/XBG_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    xgboost = xgb.XGBClassifier(n_jobs=-1,
                                objective='binary:logistic',
                                use_label_encoder=False,
                                eval_metric='logloss',
                                **best_params)

    xgboost.fit(X_train, y_train)

    # with open(filepath, 'wb') as file:
    #     pickle.dump(xgboost, file)

    return xgboost


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions
