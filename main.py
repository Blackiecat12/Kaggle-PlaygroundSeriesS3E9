# General + Data packages
import time
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from matplotlib import pyplot as plt

# ML packages
from sklearnex import patch_sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge


def remove_duplicate_instances(df: pd.DataFrame):
    """ The transform finds each set of duplicates and replaces them with a new entry with the mean objective value. """
    feature_list = df.columns.tolist()[1:-1]
    # Initialising clean data and dropping id
    cleaned_df = df.drop_duplicates(subset=feature_list, keep=False).reset_index()
    cleaned_df.drop(labels=["id", "index"], axis='columns', inplace=True)
    # Duplicate handling
    for idx, group in df[df.duplicated(feature_list, keep=False)].groupby(feature_list):
        average_instance = group.iloc[1, 1:]
        average_instance['Strength'] = group['Strength'].mean()
        pd.concat([cleaned_df, average_instance], axis='index')
    return cleaned_df


def main():
    # Enable GPU processing
    patch_sklearn()

    # Load and clean data
    full_data = pd.read_csv("Kaggle Data/train.csv")
    # full_data = remove_duplicate_instances(full_data)
    full_data.drop(columns=["id"], inplace=True)

if __name__ == "__main__":
    main()
