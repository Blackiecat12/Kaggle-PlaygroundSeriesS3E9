# General + Data packages
import time
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

# ML packages
from sklearnex import patch_sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
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


class CustomFeatures(BaseEstimator, TransformerMixin):
    """ Transformation class to add the custom features for engineering. """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['TotalComponentWeight'] = X['CementComponent'] + X['BlastFurnaceSlag'] + X['FlyAshComponent'] \
                                    + X['WaterComponent'] + X['SuperplasticizerComponent'] \
                                    + X['CoarseAggregateComponent'] + X['FineAggregateComponent']
        X['WaterCementRatio'] = X['WaterComponent'] / X['CementComponent']
        X['AggregateRatio'] = (X['CoarseAggregateComponent'] + X['FineAggregateComponent']) / X[
            'CementComponent']
        X['WaterBindingRatio'] = X['WaterComponent'] / (X['CementComponent'] + X['BlastFurnaceSlag']
                                                        + X['FlyAshComponent'])
        X['SuperPlasticizerRatio'] = X['SuperplasticizerComponent'] / (
                X['CementComponent'] + X['BlastFurnaceSlag']
                + X['FlyAshComponent'])
        X['CementAgeRelation'] = X['CementComponent'] * X['AgeInDays']
        X['AgeFactor2'] = X['AgeInDays'] ^ 2
        X['AgeFactor1/2'] = np.sqrt(X['AgeInDays'])
        return X


def create_feature_engineering_pipeline():
    """ Creates the pipeline to engineer features. """
    engineering_pipeline = Pipeline(steps=[("custom_features", CustomFeatures())])
    return engineering_pipeline


def create_preprocessing_pipeline():
    """ Creates the pipeline to run preprocessing on features. """
    preprocessing_pipeline = Pipeline(steps=[("scaling", StandardScaler())])
    return preprocessing_pipeline


def create_training_pipeline():
    """ Creates the full pipeline with feature engineering, preprocessing, and model. """
    clf = Pipeline(steps=[("feature_engineering", create_feature_engineering_pipeline()),
                          ("preprocessing", create_preprocessing_pipeline()),
                          ("model", Ridge(70))])
    return clf


def score_model_using_KFold(model, train, features, target, verbose: bool = True):
    """ Runs KFold validation on the given model.
    This code is inspired/used from https://www.kaggle.com/code/ambrosm/pss3e9-eda-which-makes-sense
    :param train: The training dataset
    :param test: The testing dataset
    """
    fold_scores = []
    kf = KFold()
    for fold_counter, (idx_train, idx_validation) in enumerate(kf.split(train)):
        X_train = train.iloc[idx_train, features]
        y_train = train.iloc[idx_train, target]
        X_validation = train.iloc[idx_validation, features]
        y_validation = train.iloc[idx_validation, target]

        model.fit(X_train, y_train)
        validation_RMSE = mean_squared_error(y_validation, model.predict(X_validation), squared=False)
        fold_scores.append(validation_RMSE)

        # Reporting
        if verbose:
            training_RMSE = mean_squared_error(y_train, model.predict(X_train), squared=False)
            print(f"Fold: {fold_counter}\tTraining RMSE: {training_RMSE:.2f}\tValidation RMSE: {validation_RMSE:.2f}")
    print(f"Average RMSE: {np.mean(fold_scores)}")


def main():
    # Enable GPU processing
    patch_sklearn()

    # Load and clean data
    full_data = pd.read_csv("Kaggle Data/train.csv")
    # full_data = remove_duplicate_instances(full_data)
    full_data.drop(columns=["id"], inplace=True)

    # Training and test sets
    train_X, test_X, train_y, test_y = train_test_split(full_data.iloc[:, :-1], full_data.iloc[:, -1], train_size=.7, random_state=300)

    # Create the pipeline
    model = create_training_pipeline()
    tic = time.perf_counter_ns()
    model.fit(train_X, train_y)
    toc = time.perf_counter_ns()

    predictions = model.predict(test_X)
    print(f"Finished Training in {(toc-tic)/1e9:.2f}s with RMSE {sklearn.metrics.mean_squared_error(test_y, predictions, squared=False)}")
    results = pd.DataFrame({"Prediction": predictions, "Actual": test_y, "Difference": predictions - test_y})

if __name__ == "__main__":
    main()
