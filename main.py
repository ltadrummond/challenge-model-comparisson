"""
main.py

The script containing data manipulation and modeling.
"""
from sklearn.metrics import plot_confusion_matrix
from utils.functions_modeling import *
from utils.functions_plotting import *
from utils.functions_pre_process import *
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

#plot_correlation(df_cleaned)

#count_plot("default_payment_next_month", df_cleaned, 'Before Resample')

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

#count_plot("default_payment_next_month", df_resampled, 'After Resample')

X_resampled = np.array(df_resampled.drop(columns='default_payment_next_month'))
y_resampled = np.array(df_resampled['default_payment_next_month'])
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled,
                                                                                            y_resampled,
                                                                                            test_size=0.3,
                                                                                            shuffle=True,
                                                                                            stratify=y_resampled)


#train_save_model(X_train_resampled, y_train_resampled, model_dict)

#compare_model_metrics(model_dict, X_train_resampled, X_test_resampled, y_train_resampled,
#                      y_test_resampled, 'after_resampling')

model_forest = pickle.load(open('models/model_random_forest', 'rb'))

#plot_confusion_matrix(model_forest, X_test_resampled, y_test_resampled)
#plt.show()

#print(model_forest.get_params())

#grid_search(model_forest, X_train_resampled, y_train_resampled)

trained_models_list = get_all_trained_models_as_list()

plot_roc_curve_auc(trained_models_list, X_test_resampled, y_test_resampled)