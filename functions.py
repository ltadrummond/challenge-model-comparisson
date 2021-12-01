"""
functions.py

The script containing the functions to be used n this project.
"""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
    f1_score, roc_curve, auc, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def compare_model_metrics(model_dict: Dict, X_train: np.array, X_test: np.array, y_train: np.array,
                          y_test: np.array) -> pd.DataFrame:
    """Do model comparison.

    This function takes as input the dictionary of the different models and the X and y split in
    train and test and it returns a data frame with the metrics and model names.
    :param model_dict:
    :param  X_train:
    :param  X_test:
    :param  y_train:
    :param  y_test:
    :return pd.DataFrame:
    """
    model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], []
    for k, v in model_dict.items():
        steps = [('scaling', StandardScaler()),
                 (k, v)]
        pipeline = Pipeline(steps)
        model_name.append(k)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        ac_score_list.append(accuracy_score(y_test, y_pred))
        p_score_list.append(precision_score(y_test, y_pred, average='macro'))
        r_score_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
        model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
        model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
        model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)
        model_comparison_df.to_csv('model_comparison.csv')
    return model_comparison_df


def resample_data(df, n_samples):
    df_majority = df[df['default_payment_next_month'] == 1]
    df_minority = df[df['default_payment_next_month'] == 0]
    df_majority_downsampled = resample(df_majority, n_samples=n_samples)
    df_minority_upsampled = resample(df_minority, n_samples=n_samples)
    df_resampled = pd.concat([df_majority_downsampled, df_minority_upsampled])
    return df_resampled


def plot_importance(model_fit, X_train: pd.DataFrame) -> None:
    """Plot count of the target variable.

    :param model_fit:
    :param  X_train:
    :return plt.Axes:
    """
    importance = model_fit.feature_importances_
    plt.bar(range(len(importance)), importance)
    plt.xticks(range(len(importance)), X_train.columns, rotation=70)
    plt.ylabel('importance coefficient')
    plt.title("Importance level by features")
    plt.show


def plot_correlation(df: pd.DataFrame) -> None:
    """Plot correlation between the features.

    :param  df:
    :return None:
    """
    plt.figure()
    ax = sns.heatmap(df.corr(), linewidths=.5)
    plt.show()


def count_plot(x: str, df: pd.DataFrame) -> None:
    """Plot count of the target variable.

    :param x:
    :param  df:
    :return None:
    """
    plt.figure()
    s = sns.countplot(x=x, data=df, palette="Paired")
    s.set_title("Number of people who have a faulty payment or not")
    s.set(xlabel='faulty payment', ylabel='count')
    plt.show()



