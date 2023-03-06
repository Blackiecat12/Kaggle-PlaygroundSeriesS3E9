# Data handling
import numpy as np
import pandas as pd

# Data Investigation
# import sklearn

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns


def report_basic_info(df: pd.DataFrame):
    print(df.info())
    print(df.describe())


def main():
    df = pd.read_csv("Kaggle Data/train.csv")
    report_basic_info(df)


if __name__ == "__main__":
    main()
