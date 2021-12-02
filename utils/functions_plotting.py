"""
functions_plotting.py

The script containing functions to plot.
"""
from typing import List
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot
from sklearn.model_selection import ShuffleSplit, learning_curve
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


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


def count_plot(x: str, df: pd.DataFrame, before_or_after_resample: str) -> None:
    """Plot count of the target variable.

    :param before_or_after_resample:
    :param x:
    :param  df:
    :return None:
    """
    plt.figure()
    s = sns.countplot(x=x, data=df, palette="Paired")
    s.set_title(f"Distribution Default Payments {before_or_after_resample}")
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


def get_all_trained_models_as_list():
    path = os.getcwd()
    os.chdir('./models')
    all_files = os.listdir()
    models_list = []
    for file in os.listdir():
        if not file.endswith('.csv') and not file.endswith('cent'):
            model = pickle.load(open(file, 'rb'))
            models_list.append(model)
    print(models_list)
    return models_list


def plot_roc_curve_auc(models_list: List, X_test: np.array, y_test: np.array) -> None:
    """Calculate AUC and plot ROC.

    :param models_list:
    :param X_test:
    :param y_test:
    :return None:
    """
    plt.subplots(figsize=(8, 6))
    for model in models_list:
        roc = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, roc)
        plt.plot(fpr, tpr, label=model[1])
    x = np.linspace(0, 1, num=50)
    plt.plot(x, x, color='lightgrey', linestyle='--', marker='', lw=2, label='random guess')
    plt.legend(fontsize=14)
    plt.xlabel('False positive rate', fontsize=18)
    plt.ylabel('True positive rate', fontsize=18)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title('ROC Model Comparison')
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