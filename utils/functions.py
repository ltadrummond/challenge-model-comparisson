"""
functions.py

The script containing the functions to be used n this project.
"""
import pickle
from typing import Dict
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV, ShuffleSplit, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
    f1_score, roc_curve, auc, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
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


def grid_search(model: Pipeline, X_train: np.array, y_train: np.array) -> None:
    """Perform grid search and create a data frame with it's values.

    This function performs grid search with Random Forest classifier. Then it stores the best parameters and
    the accuracy in a dataframe.
    :param model:
    :param X_train:
    :param y_train:
    :return None:
    """
    model_name,  grid_accuracy, best_parameters = ['random_forest'], [], []
    parameters = {'random_forest__n_estimators': [50, 100, 150, 200],
                  'random_forest__min_samples_split': [2, 3, 4],
                  'random_forest__criterion': ['gini', 'entropy'],
                  'random_forest__max_features': ['auto', 'sqrt', 'log2']}
    gs_clf = GridSearchCV(model, parameters, cv=4, scoring='accuracy')
    gs_clf = gs_clf.fit(X_train, y_train)
#    print("Best accuracy = %.3f%%" %((gs_clf.best_score_)*100.0))
#    print("Best parameters are : ")
    model_name.append('random_forest')
    best_parameters.append(gs_clf.best_params_)
    grid_accuracy.append(gs_clf.best_score_)
    grid_df = pd.DataFrame([model_name, grid_accuracy, best_parameters]).T
    grid_df.columns = ['model_name', 'grid_accuracy', 'best_parameters']
    grid_df.to_csv('models/model_performance_after_grid.csv')


parameters_model_forest = {'verbose': False,
                           'scaling__copy': True, 'scaling__with_mean': True, 'scaling__with_std': True,
                           'random_forest__bootstrap': True, 'random_forest__ccp_alpha': 0.0,
                           'random_forest__class_weight': None, 'random_forest__criterion': 'gini',
                           'random_forest__max_depth': None, 'random_forest__max_features': 'auto',
                           'random_forest__max_leaf_nodes': None,
                           'random_forest__max_samples': None, 'random_forest__min_impurity_decrease': 0.0,
                           'random_forest__min_samples_leaf': 1, 'random_forest__min_samples_split': 2,
                           'random_forest__min_weight_fraction_leaf': 0.0, 'random_forest__n_estimators': 100,
                           'random_forest__n_jobs': None, 'random_forest__oob_score': False,
                           'random_forest__random_state': None, 'random_forest__verbose': 0,
                           'random_forest__warm_start': False}


def train_save_model(X_train: np.array, y_train: np.array, models_dictionary: Dict) -> None:
    """Train and save the models in a pickle file

    :param X_train:
    :param y_train
    :param models_dictionary:
    :return None:
    """
    for key, value in models_dictionary.items():
        steps = [('scaling', StandardScaler()), (key, value)]
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)
        print(pipeline.get_params())
        pickle.dump(pipeline, open(f'models/model_{key}', 'wb'))


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
    ax = sns.heatmap(df.corr(), linewidths=.5, cmap="YlGnBu")
    ax.set_title('Correlation Between Features')
    plt.show()


def count_plot(x: str, df: pd.DataFrame, before_or_afte_resample: str) -> None:
    """Plot count of the target variable.

    :param x:
    :param  df:
    :return None:
    """
    plt.figure()
    s = sns.countplot(x=x, data=df, palette="Paired")
    s.set_title(f"Distribution Default Payments {before_or_afte_resample}")
    s.set(xlabel='Default payment', ylabel='Count')
    plt.show()


def evaluate_and_plot_train_test(model: Pipeline, X_train: np.array, X_test: np.array, y_train: np.array,
                                 y_test: np.array) -> None:
    """This function aims to detect over-fitting.

    :param model:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return None:
    """
    train_scores, test_scores = [], []
    values = [i for i in range(1, 51)]
    # evaluate a decision tree for each depth
    for i in values:
        model = model
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_scores.append(train_acc)
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_scores.append(test_acc)
        # summarize progress
        print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))
    plt.fig()
    pyplot.plot(values, train_scores, '-o', label='Train')
    pyplot.plot(values, test_scores, '-o', label='Test')
    pyplot.legend()
    pyplot.show()


def plot_roc_curve_auc(model: Pipeline, X_test: np.array, y_test: np.array) -> None:
    """Calculate AUC and plot ROC.

    :param model:
    :param X_test:
    :param y_test:
    :return None:
    """
    roc = model.predict_proba(X_test)[:, 1]
    plt.subplots(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(y_test, roc)
    plt.plot(fpr, tpr)
    x = np.linspace(0, 1, num=50)
    plt.fig()
    plt.plot(x, x, color='lightgrey', linestyle='--', marker='', lw=2, label='random guess')
    plt.legend(fontsize=14)
    plt.xlabel('False positive rate', fontsize=18)
    plt.ylabel('True positive rate', fontsize=18)
    plt.title(f'ROC Curve/ AUC = {auc(fpr, tpr)}')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.show()


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(0.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                                          train_sizes=train_sizes, return_times=True,)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color="g",)
    axes[0].plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")
    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    plt.show()
    return plt
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    title = "Learning Curves (Naive Bayes)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = KNeighborsClassifier()

    plot_learning_curve(estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = Random
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)
