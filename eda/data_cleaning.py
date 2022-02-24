"""
Cleaning method. Implements a very basic cleaning routine. Features with low data
quality, such as capital-gain and -loss are dropped.
"""
import pandas as pd
import numpy as np


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
    data_frame = pd.read_csv("../data/census.csv", skipinitialspace=True)
    clean_df = clean_data(data_frame)
    clean_df.to_csv("../data/census_clean_v1.csv", index=False)