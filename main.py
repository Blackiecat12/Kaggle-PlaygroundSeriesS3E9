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


def plot_partial_dependence(model, full_data, features):
    # Partial Dependence Display
    tic = time.perf_counter_ns()
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    plt.suptitle('Partial Dependence', y=1.0)
    PartialDependenceDisplay.from_estimator(model, full_data.loc[:, features],
                                            features,
                                            pd_line_kw={"color": "red"},
                                            ice_lines_kw={"color": "blue"},
                                            kind='both',
                                            ax=axs.ravel()[:len(features)])
    plt.tight_layout(h_pad=0.3, w_pad=0.5)
    toc = time.perf_counter_ns()
    print(f"Partial Dependence Display done in {(toc - tic) / 1e9:.2f}s")
    plt.show()


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


def create_training_pipeline(model):
    """ Creates the full pipeline with feature engineering, preprocessing, and model.
     :param model: The model to train
     :return training pipeline
     """
    clf = Pipeline(steps=[("feature_engineering", create_feature_engineering_pipeline()),
                          ("preprocessing", create_preprocessing_pipeline()),
                          ("model", model)])
    return clf


def score_model_using_KFold(model, train, features, target, verbose: bool = True):
    """ Runs KFold validation on the given model.
    This code is inspired/used from https://www.kaggle.com/code/ambrosm/pss3e9-eda-which-makes-sense
    :param model: The model to test
    :param train: The training dataset
    :param features: The features to train on
    :param target: The prediction target
    :param verbose: Output fold RMSE values
    """
    fold_scores = []
    kf = KFold()
    tic = time.perf_counter_ns()
    for fold_counter, (idx_train, idx_validation) in enumerate(kf.split(train)):
        X_train = train.loc[idx_train, features]
        y_train = train.loc[idx_train, target]
        X_validation = train.loc[idx_validation, features]
        y_validation = train.loc[idx_validation, target]

        model.fit(X_train, y_train)
        validation_RMSE = mean_squared_error(y_validation, model.predict(X_validation), squared=False)
        fold_scores.append(validation_RMSE)

        # Reporting
        if verbose:
            training_RMSE = mean_squared_error(y_train, model.predict(X_train), squared=False)
            print(f"Fold: {fold_counter}\tTraining RMSE: {training_RMSE:.2f}\tValidation RMSE: {validation_RMSE:.2f}")
    toc = time.perf_counter_ns()
    print(f"Model {str(model.steps[-1][-1]).split('(')[0]}\n\t"
          f"Average RMSE: {np.mean(fold_scores)}\n\t"
          f"Time Taken: {(toc-tic)/1.e9:.2f}s")


def main():
    # Enable GPU processing
    patch_sklearn()

    # Load and clean data
    full_data = pd.read_csv("Kaggle Data/train.csv")
    # full_data = remove_duplicate_instances(full_data)

    # Create the pipeline
    model = create_training_pipeline(Ridge(70))
    score_model_using_KFold(model, full_data, full_data.columns[1:-1], full_data.columns[-1], False)


if __name__ == "__main__":
    main()
