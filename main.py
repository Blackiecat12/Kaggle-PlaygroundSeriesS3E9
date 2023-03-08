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


def main():
    # Enable GPU processing
    patch_sklearn()


if __name__ == "__main__":
    main()
