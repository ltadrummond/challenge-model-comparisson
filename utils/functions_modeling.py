"""
functions_modeling.py

The script containing the functions to do tasks related to modeling.
"""
import os
import pickle
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
    f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def get_all_trained_models_as_list():
    path = os.getcwd()
    os.chdir('./models')
    all_files = os.listdir()
    models_list = []
    for file in os.listdir():
        if not file.endswith('.csv'):
            model = pickle.load(open(file, 'rb'))
            models_list.append(model)
    print(models_list)
    return models_list


