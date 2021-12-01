"""
modeling.py

The script containing data manipulation and modeling.
"""

from utils.functions import *
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


df = pd.read_csv('data/UCI_Credit_Card.csv')
#display_logs(df)

df_cleaned = preprocess_df(df)

X = np.array(df_cleaned.drop(columns='default_payment_next_month'))
y = np.array(df_cleaned['default_payment_next_month'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)

model_dict = {'dummy': DummyClassifier(),
              'Stochastic Gradient Descent': SGDClassifier(),
              'random_forest': RandomForestClassifier(),
              'decision_tree': DecisionTreeClassifier(),
              'ada_boost': AdaBoostClassifier(),
              'logistic_regression': LogisticRegression(),
              'k_nearest_neighbor': KNeighborsClassifier()}

#compare_model_metrics(model_dict, X_train, X_test, y_train, y_test, 'before_resampling')

df_resampled = resample_data(df_cleaned, 17000)
X_resampled = np.array(df_resampled.drop(columns='default_payment_next_month'))
y_resampled = np.array(df_resampled['default_payment_next_month'])
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled,
                                                                                            y_resampled,
                                                                                            test_size=0.3,
                                                                                            shuffle=True,
                                                                                            stratify=y_resampled)

pickle.dump(X_train_resampled, open('utils/X_train_resampled', 'wb'))

#train_save_model(X_train_resampled, y_train_resampled, model_dict)
#compare_model_metrics(model_dict, X_train_resampled, X_test_resampled, y_train_resampled,
#                      y_test_resampled, 'after_resampling')


model_forest = pickle.load(open('models/model_random_forest', 'rb'))

print(model_forest.get_params())
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