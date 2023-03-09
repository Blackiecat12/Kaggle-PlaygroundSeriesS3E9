# Data manip/vis
import math
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ML tasks
from sklearnex import patch_sklearn
from sklearn.ensemble import RandomForestRegressor


def report_basic_info(df: pd.DataFrame):
    """ Prints the information of the columns in the dataframe, then prints the statistical variation of the
    columns. """
    print(df.info())
    for col in df.columns:
        print("-------")
        print(df[col].describe())


def report_duplicated(df: pd.DataFrame):
    """ Prints the number duplicated instances over all columns, then excludes id, then excludes strength. """
    df_size = len(df)
    total_duped = sum(df.duplicated())
    id_duped = sum(df.duplicated(subset=df.columns.to_list()[1:]))
    id_str_duped = sum(df.duplicated(subset=df.columns.to_list()[1:-1]))
    print(f"Full dupes: {total_duped} ({total_duped / df_size * 100:.1f}%)\n"
          f"Without ID dupes: {id_duped} ({id_duped / df_size * 100:.1f}%)\n"
          f"Feature dupes: {id_str_duped} ({id_str_duped / df_size * 100:.1f}%)")


def plot_duplicated_distributions(df: pd.DataFrame):
    """ Plots the distribution of the times a row is duplicated. Then plots the distribution of the ranges that the
    strength of that duplicated row has. """
    feature_list = df.columns.tolist()[1:-1]
    df_duplicated_count = df.groupby(feature_list, as_index=False).size()
    df_duplicated_count = df_duplicated_count[df_duplicated_count['size'] != 1]
    v_counts = df_duplicated_count['size'].value_counts()
    sns.barplot(x=v_counts.keys().to_list(), y=v_counts.values)
    plt.xlabel("Times repeated")
    plt.ylabel("Row Count")
    plt.title("Distribution of times a row is repeated")
    plt.show()
    # Grouping into the duplicates
    groups = {}
    ranges = []
    for idx, group in df[df.duplicated(feature_list, keep=False)].groupby(feature_list):
        groups[idx] = group
        ranges.append(group['Strength'].max() - group['Strength'].min())
    sns.violinplot(ranges)
    plt.ylabel("Range of Strength in duplicated rows")
    plt.title("Distribution of strength ranges between the duplicated rows")
    plt.show()


def analyse_feature_distributions(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    """ Plots the distribution of the feature values.
    This code is inspired/used from https://www.kaggle.com/code/ambrosm/pss3e9-eda-which-makes-sense
    :param train: The training dataframe
    :param test: The test dataframe
    :param features: The feature columns to plot
    """
    # Set up the plot
    plot_width = 3
    plot_height = math.ceil(len(features)/plot_width)
    _, axs = plt.subplots(plot_width, plot_height, figsize=(12, 10))
    ax_list = axs.ravel()

    # Iterate the features and axes to create the full dist
    for feature, ax in zip(features, ax_list):
        if feature in test:
            plot_data = pd.concat([train[feature], test[feature]], axis='columns')
            plot_data.columns = ["Train", "Test"]
        else:
            plot_data = train[feature]
            plot_data.columns = ["Train"]
        plot_feature_distribution(plot_data, feature, ax)
    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    plt.show()


def plot_feature_distribution(data: pd.DataFrame, feature: str, ax: plt.Axes):
    """ Plots a single feature distribution with a histogram and returns the plot.
    :param data: The dataframe to plot
    :param feature: The feature name
    :param ax: Where to plot
    """
    bin_number = 40
    if len(np.unique(data)) < bin_number:
        sns.histplot(data=data, stat='density', ax=ax, line_kws={'alpha': 0.5})
    else:
        sns.histplot(data=data, stat='density', ax=ax, line_kws={'alpha': 0.5})
    ax.set_title(feature)


def analyse_correlations(data: pd.DataFrame):
    """ Takes the correlation and plots as a heatmap.
    This code is inspired/used from https://www.kaggle.com/code/ambrosm/pss3e9-eda-which-makes-sense
    :param data: The data to correlate
    """
    corr_matrix = data.corr()
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr_matrix, linewidth=0.1, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='PiYG', center=0)
    plt.xticks(fontsize='x-small', rotation=40)
    plt.show()


def add_engineered_features(df: pd.DataFrame):
    """ Adds various features to the input df. """
    # TotalComponentWeight
    df['TotalComponentWeight'] = df['CementComponent'] + df['BlastFurnaceSlag'] + df['FlyAshComponent'] \
                                 + df['WaterComponent'] + df['SuperplasticizerComponent'] \
                                 + df['CoarseAggregateComponent'] + df['FineAggregateComponent']
    df['WaterCementRatio'] = df['WaterComponent'] / df['CementComponent']
    df['AggregateRatio'] = (df['CoarseAggregateComponent'] + df['FineAggregateComponent']) / df['CementComponent']
    df['WaterBindingRatio'] = df['WaterComponent'] / (df['CementComponent'] + df['BlastFurnaceSlag']
                                                      + df['FlyAshComponent'])
    df['SuperPlasticizerRatio'] = df['SuperplasticizerComponent'] / (df['CementComponent'] + df['BlastFurnaceSlag']
                                                          + df['FlyAshComponent'])
    df['CementAgeRelation'] = df['CementComponent'] * df['AgeInDays']
    df['AgeFactor2'] = df['AgeInDays'] ^ 2
    df['AgeFactor1/2'] = np.sqrt(df['AgeInDays'])
    return df

def run_feature_selection(df: pd.DataFrame, method: str = 'RFC'):
    """ Prints the importance of the features to the predicted value via the given method. """
    X = df.drop(['id', 'Strength'], axis=1)
    y = df['Strength']
    start_time = time.perf_counter_ns()
    rf = RandomForestRegressor()
    rf.fit(X, y)
    importances = rf.feature_importances_
    end_time = time.perf_counter_ns()
    sorted_idx = importances.argsort()
    print(f"Feature Importance via {method} calculated in {(end_time - start_time)/1e9:.3f}s")
    print(pd.Series(importances[sorted_idx], index=X.columns[sorted_idx]).sort_values(ascending=False)[:10])

    corr_matrix = df.corr()
    print("Correlation")
    print(corr_matrix['Strength'].sort_values(ascending=False)[1:11])


def main():
    """ Main function to run the various EDA tasks. """
    df = pd.read_csv("Kaggle Data/train.csv")
    report_basic_info(df)
    report_duplicated(df)
    plot_duplicated_distributions(df)
    # Feature engineering
    df = add_engineered_features(df)

    # Setting up sklearn with GPU
    patch_sklearn()

    # Feature importance
    run_feature_selection(df)
    corr_matrix = df.corr()
    train_df = pd.read_csv("Kaggle Data/train.csv")
    test_df = pd.read_csv("Kaggle Data/test.csv")
    # Info
    # report_basic_info(df)
    # report_duplicated(df)
    # plot_duplicated_distributions(df)
    # analyse_feature_distributions(df, pd.read_csv("Kaggle Data/test.csv"), df.columns[1:])
    # analyse_correlations(df.iloc[:, 1:])


if __name__ == "__main__":
    main()
