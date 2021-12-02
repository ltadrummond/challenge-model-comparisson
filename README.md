# Model Comparisson Project

<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/readme_intro_pic.png">

This project aims to classify default payments in credit cards. The dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. The dataset can also be found on klaggle in the following link: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset


# Installation
* Python 3.7
* Pycharm


# Libaries
* Pandas
* Numpy
* Matplotlib
* sklearn
* Pickle
* Seaborn



# Files


| File                        | Description                                                     |
|-----------------------------|-----------------------------------------------------------------|
| models                | Folder containing trained models and csv files with model metrics comparisson. |
| utils             | Folder containing python files with functions and visualization.|
|.gitignore  | Text file containing names of the files to be ignored in the remote repository. |
| visuals            | Folder containing plots that are interesting and helped to bring insight.  |
| modeling.py            | Python file containing code to set up the model. |



# Method
| Step                      | Description                                                     |
|-----------------------------|-----------------------------------------------------------------|
| Analyzing the data                | null values, data types, duplicates, unique values by column, description of data statistics, visualize tail and head|
| Visualizing the data          | correlation plot, target distribution plot|
|Preporcessing  |  dropping column "ID", renaming target feature, normalizing, train and test split, resampling target feature|
| Modeling (1st)           | train on not resampled train set, for different models, and compare their metrics on test set  |
| Modeling (2nd)             | train on resampled train set, for different models, and compare their metrics on test set - models saved as pickle file|
| Visualizing the models performance       | plot the ROC curve for the different models on test set|
| Modeling (3nd)             | perform grid search on best performing model (Random Forest) and display it's confusion matrix and metrics on test set |



# Visuals

<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/correlation_feautures.png">
    

## Before Resampling:

* Distribution of the target feature:
<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/target_features_distribution.png">


* Comparisson of the performance of different algorythms:
<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/tabel_metrics_before_resampling.png">


## After Resampling:

* Distribution of the target feature:

<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/target_features_after_resampling.png">


* Comparisson of the performance of different algorythms:
<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/metrics_after_resampling.png">


* ROC for the different models:
<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/ROC_model_comparison.png">



## Random Forest After grid search:

* Confusion Matrix Random Forest
<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/confusion_matrix.png">

* Accuracy Random Forest
<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/model_after_grid.png">

# Conclusion
To achieve a better performing model, normalizing and resampling were two important preprocessing steps. Also being able to compare classifiers and tunning the hyperparameters, are needed to be taken into consideration.
The best performed model was Random Forest with an accuracy of 96% on the test set.

Accuracy was the chosen metric in this case, since it was possible to evaluate the model on a balanced dataset.



# Time-line
30-11-2021 to 02-12-2021

# Contributors
| Name                  | Github                                 |
|-----------------------|----------------------------------------|
|Leonor Drummonnd      | https://github.com/ltadrummond              |
