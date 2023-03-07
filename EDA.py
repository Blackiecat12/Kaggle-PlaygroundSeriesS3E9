# Data manip/vis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# ML tasks
from sklearn.ensemble import RandomForestClassifier


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


def main():
    """ Main function to run the various EDA tasks. """
    df = pd.read_csv("Kaggle Data/train.csv")
    report_basic_info(df)
    report_duplicated(df)
    plot_duplicated_distributions(df)
    # Feature engineering
    df = add_engineered_features(df)

if __name__ == "__main__":
    main()
