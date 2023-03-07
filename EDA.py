import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def report_basic_info(df: pd.DataFrame):
    """ Prints the information of the columns in the dataframe, then prints the statistical variation of the
    columns. """
    print(df.info())
    print(df.describe())


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
    """ Plots the distributions of the duplicated values. """
    df_duplicated_count = df.groupby(df.columns.tolist()[1:-1], as_index=False).size()
    df_duplicated_count = df_duplicated_count[df_duplicated_count['size'] != 1]
    v_counts = df_duplicated_count['size'].value_counts()
    sns.barplot(x=v_counts.keys().to_list(), y=v_counts.values)
    plt.xlabel("Times repeated")
    plt.ylabel("Row Count")
    plt.title("Distribution of times a row is repeated")
    plt.show()


def main():
    """ Main function to run the various EDA tasks. """
    df = pd.read_csv("Kaggle Data/train.csv")
    report_basic_info(df)
    report_duplicated(df)
    plot_duplicated_distributions(df)

if __name__ == "__main__":
    main()
