"""
modeling.py

The script containing data manipulation, plotting and modeling
"""

from functions import *
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv('data/UCI_Credit_Card.csv')

display_logs(df)
df_cleaned = preprocess_df(df)

model_dict = {'dummy': DummyClassifier(),
              'Stochastic Gradient Descent': SGDClassifier(),
              'random_forest': RandomForestClassifier(),
              'decision_tree': DecisionTreeClassifier(),
              'ada_boost': AdaBoostClassifier(),
              'logistic_regression': LogisticRegression(),
              'k_nearest_neighbor': KNeighborsClassifier()}


X = np.array(df_cleaned.drop(columns='default_payment_next_month'))
y = np.array(df_cleaned['default_payment_next_month'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y)

#model_comparison = compare_model_metrics(model_dict, X_train, X_test, y_train, y_test)

df_resampled = resample_data(df_cleaned, 17000)

print(df_resampled.default_payment_next_month.value_counts())


