"""
functions_pre_process.py

The script containing functions to pre-process the data.
"""

import pandas as pd
from sklearn.utils import resample


def display_logs(df: pd.DataFrame) -> None:
    """Get quick information over the dataframe.

    This function prints a couple of data frame information like name of the columns, data types,
    description, unique values.
    :param df:
    :return:
    """
    print(df.head)
    print('the data types are: ', df.dtypes)
    print('the sum of the null values is: ', df.isnull().sum())
    print('the names of the columns are: ', df.columns)
    print('the description of the data is: ', df.describe())
    print('the shape of the data is: ', df.shape)
    for column in df.columns.tolist():
        print(str(column))
        print(df[column].unique())


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """This function drops the index and renames default payment columns.
    :param df:
    :return pd.DataFrame:
    """
    print('shape before dropping the id: ', df.shape)
    df = df.rename(columns={"default.payment.next.month": "default_payment_next_month"})
    df = df.drop(columns='ID')
    print('shape after dropping the id: ', df.shape)
    return df


def resample_data(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Resample the target feature

    This function takes into consideration the over-represented class and the under-represented class,
    and resamples them to bring balance to the target feature representation
    :param df:
    :param  n_samples:
    :return pd.DataFrame:
    """
    df_majority = df[df['default_payment_next_month'] == 1]
    df_minority = df[df['default_payment_next_month'] == 0]
    df_majority_downsampled = resample(df_majority, n_samples=n_samples)
    df_minority_upsampled = resample(df_minority, n_samples=n_samples)
    df_resampled = pd.concat([df_majority_downsampled, df_minority_upsampled])
    return df_resampled