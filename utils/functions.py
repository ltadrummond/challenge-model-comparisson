"""
functions.py

The script containing the functions to be used n this project.
"""
import pickle
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


def compare_model_metrics(models_dictionary: Dict, X_train: np.array, X_test: np.array, y_train: np.array,
                          y_test: np.array, before_or_after_resampling: str) -> pd.DataFrame:
    """Do model comparison.

    This function takes as input the dictionary of the different models and the X and y split in
    train and test and it returns a data frame with the metrics and model names.
    :param models_dictionary:
    :param  X_train:
    :param  X_test:
    :param  y_train:
    :param  y_test:
    :param  before_or_after_resampling:
    :return pd.DataFrame:
    """
    model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], []
    for k, v in models_dictionary.items():
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
        model_comparison_df.to_csv(f'models/model_comparison_{before_or_after_resampling}.csv')
    return model_comparison_df


def grid_search(model, X_train, y_train):
    parameters = {'estimator__n_estimators': [20, 50, 70, 100],
                  'estimator__learning_rate': [0.01, 0.1, 0.2, 2],
                  'estimator__algorithm': ['SAMME', 'SAMME.R']}
    gs_clf = GridSearchCV(model, parameters, cv=4, n_jobs=-1, scoring='f1_macro')
    gs_clf = gs_clf.fit(X_train, y_train)
    print("Best f1_macro = %.3f%%" %((gs_clf.best_score_)*100.0))
    print("Best parameters are : ")
    print(gs_clf.best_params_)


def train_save_model(X_train: np.array, y_train: np.array, models_dictionary: Dict) -> None:
    for k, v in models_dictionary.items():
        steps = [('scaling', StandardScaler()),
             (k, v)]
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)
        print(pipeline.get_params())
        pickle.dump(pipeline, open(f'models/model_{k}', 'wb'))


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


def plot_importance(model_fit, X_train: np.array) -> None:
    """Plot count of the target variable.

    :param model_fit:
    :param  X_train:
    :return plt.Axes:
    """
    plt.figure()
    importance = model_fit.steps[1][1].feature_importances_
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



