"""
visualizations.py

The script containing visuals over the data.
"""
from sklearn.model_selection import train_test_split

from functions import *
from sklearn.metrics import plot_confusion_matrix

df = pd.df = pd.read_csv('../data/UCI_Credit_Card.csv')
df_cleaned = preprocess_df(df)

plot_correlation(df_cleaned)

count_plot("default_payment_next_month", df_cleaned, 'Before Resample')

df_resampled = resample_data(df_cleaned, 17000)

count_plot("default_payment_next_month", df_resampled, 'After Resample')

X_resampled = np.array(df_resampled.drop(columns='default_payment_next_month'))
y_resampled = np.array(df_resampled['default_payment_next_month'])
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled,
                                                                                            y_resampled,
                                                                                            test_size=0.3,
                                                                                            shuffle=True,
                                                                                            stratify=y_resampled)

model_forest = pickle.load(open('../models/model_random_forest', 'rb'))

plot_confusion_matrix(model_forest, X_test_resampled, y_test_resampled)
plt.show()