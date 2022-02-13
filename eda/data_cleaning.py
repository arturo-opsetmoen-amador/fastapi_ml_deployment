"""
Cleaning method
"""
import pandas as pd


def clean_data(df):
    df.replace({'?': None}, inplace=True)
    df.dropna(inplace=True)
    df.drop("capital-gain", axis="columns", inplace=True)
    df.drop("capital-loss", axis="columns", inplace=True)
    df.drop("education-num", axis="columns", inplace=True)
    df.drop("fnlgt", axis="columns", inplace=True)
    return df


if __name__ == "__main__":
    df = pd.read_csv("../data/census.csv", skipinitialspace=True)
    clean_df = clean_data(df)
    df.to_csv("../data/census_clean_v1.csv", index=False)