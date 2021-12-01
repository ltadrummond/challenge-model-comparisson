"""
visualizations.py

The script containing visuals over the data.
"""
from functions import *

df = pd.df = pd.read_csv('../data/UCI_Credit_Card.csv')
df_cleaned = preprocess_df(df)

plot_correlation(df_cleaned)

count_plot("default_payment_next_month", df_cleaned)

df_resampled = resample_data(df_cleaned, 17000)

count_plot("default_payment_next_month", df_resampled)

