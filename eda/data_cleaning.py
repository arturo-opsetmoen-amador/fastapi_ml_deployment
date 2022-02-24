"""
Cleaning method. Implements a very basic cleaning routine. Features with low data
quality, such as capital-gain and -loss are dropped.
"""

import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from ml_training.train_model import read_data


def clean_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    # Replace string ?
    data_frame.replace({'?': np.nan}, inplace=True)
    data_frame.dropna(inplace=True)
    # Drop features with low data quality. This schame needs to be implemented also during
    # predictions.
    data_frame.drop("capital-gain", axis="columns", inplace=True)
    data_frame.drop("capital-loss", axis="columns", inplace=True)
    data_frame.drop("education-num", axis="columns", inplace=True)
    data_frame.drop("fnlgt", axis="columns", inplace=True)
    return data_frame


if __name__ == "__main__":
    data = read_data("data/census.csv")
    data = pd.read_csv("data/census.csv", skipinitialspace=True)
    clean_df = clean_data(data)
    clean_df.to_csv("data/census_clean_v1.csv", index=False)
