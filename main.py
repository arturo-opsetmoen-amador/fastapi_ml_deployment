"""
# TODO: ADD DOCSTRING
Author: Arturo Opsetmoen Amador
Date: February, 2022.
"""

import os
from fastapi import FastAPI, Body
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import *
import pandas as pd
import joblib
import sys
sys.path.insert(1, './ml_training/ml')
import model, data

# from ml_training.ml.model import *
# from ml_training.ml.data import *

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI(
    title="API for classification of salaries based on census data.",
    description="Prediction interface for XGBoost model",
    version="0.1.0"
    )


class XGBoostInput(BaseModel):
    """
    # TODO: ADD DOCSTRING
    """
    age: int = Field(..., example=38)
    workclass: Literal[
        "State-gov",
        "Self-emp-not-inc",
        "Private",
        "Federal-gov",
        "Local-gov",
        "Self-emp-inc",
        "Without-pay",
        "Never-worked"
        ] = Field(..., example="Private")
    fnlgt: int = Field(..., example=71211)
    education: Literal[
        "Bachelors",
        "HS-grad",
        "11th",
        "Masters",
        "9th",
        "Some-college",
        "Assoc-acdm",
        "7th-8th",
        "Doctorate",
        "Assoc-voc",
        "Prof-school",
        "5th-6th",
        "10th",
        "Preschool",
        "12th",
        "1st-4th"
        ] = Field(..., example="Doctorate")
    education_num: float = Field(..., example=15, alias="education-num")
    marital_status: Literal[
        "Never-married",
        "Married-civ-spouse",
        "Divorced",
        "Married-spouse-absent",
        "Separated",
        "Married-AF-spouse",
        "Widowed"
        ] = Field(..., example="Never-married", alias="marital-status")
    occupation: Literal[
        "Adm-clerical",
        "Exec-managerial",
        "Handlers-cleaners",
        "Prof-specialty",
        "Other-service",
        "Sales",
        "Transport-moving",
        "Farming-fishing",
        "Machine-op-inspct",
        "Tech-support",
        "Craft-repair",
        "Protective-serv",
        "Armed-Forces",
        "Priv-house-serv"
        ] = Field(..., example="Exec-managerial")
    relationship: Literal[
        "Not-in-family",
        "Husband",
        "Wife",
        "Own-child",
        "Unmarried",
        "Other-relative"
        ] = Field(..., example="Unmarried")
    race: Literal[
        "White",
        "Black",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other"
        ] = Field(..., example="Other")
    sex: Literal["Male",
        "Female"
        ] = Field(..., example="Male")
    capital_gain: float = Field(..., example=293485, alias="capital-gain")
    capital_loss: float = Field(..., example= 1, alias="capital-loss")
    hours_per_week: float = Field(..., example=60, alias='hours-per-week')
    native_country: Literal[
        "United-States",
        "Cuba",
        "Jamaica",
        "India",
        "Mexico",
        "Puerto-Rico",
        "Honduras",
        "England",
        "Canada",
        "Germany",
        "Iran",
        "Philippines",
        "Poland",
        "Columbia",
        "Cambodia",
        "Thailand",
        "Ecuador",
        "Laos",
        "Taiwan",
        "Haiti",
        "Portugal",
        "Dominican-Republic",
        "El-Salvador",
        "France",
        "Guatemala",
        "Italy",
        "China",
        "South",
        "Japan",
        "Yugoslavia",
        "Peru",
        "Outlying-US(Guam-USVI-etc)",
        "Scotland",
        "Trinadad&Tobago",
        "Greece",
        "Nicaragua",
        "Vietnam",
        "Hong",
        "Ireland",
        "Hungary",
        "Holand-Netherlands",
        ] = Field(..., 'native-country')


class XGBOut(BaseModel):
    """
    TODO: ADD DOCSTRING
    """
    prediction: str


@app.get("/")
async def welcome():
    return {"Welcome": "Welcome to the salary XGBoost predictor web service."}

@app.post("/predict", response_model=XGBOut, status_code=200)
async def xgb_predict(input_data: XGBoostInput):
    """

    Parameters
    ----------
    input_data

    Returns
    -------

    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
        ]
    data_input = pd.DataFrame.from_dict([input_data.dict(by_alias=True)])



