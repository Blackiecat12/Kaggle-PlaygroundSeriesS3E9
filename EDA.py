import pandas as pd


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


def main():
    """ Main function to run the various EDA tasks. """
    df = pd.read_csv("Kaggle Data/train.csv")
    report_basic_info(df)
    report_duplicated(df)


if __name__ == "__main__":
    main()
