# Model Comparisson Project

<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/readme_intro_pic.png">

This project aims to classify default payments in credit cards. The dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. The dataset can also be found on klaggle in the following link: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset

To be able to do this classification task, the ID column was dropped and the target feature was resampled. Various algorythms were compared in order to achieve a better performing model. Also it was taken into consideration the possibility of overfitting.


# Installation:
* Python 3.7
* Pycharm


# Libaries:
* Pandas
* Numpy
* Matplotlib
* sklearn
* Pickle
* Seaborn



# Files:


| File                        | Description                                                     |
|-----------------------------|-----------------------------------------------------------------|
| models                | Folder containing trained models and csv files with model metrics comparisson. |
| utils             | Folder containing python files with functions and visualization.|
|.gitignore  | Text file containing names of the files to be ignored in the remote repository. |
| visuals            | Folder containing plots that are interesting and helped to bring insight.  |
| modeling.py            | Python file containing code to set up the model. |


# Visuals

<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/correlation_feautures.png">


## Before resampling:

* Distribution of the target feature:
<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/target_features_distribution.png">


* Comparisson of the performance of different algorythms:



## After resmpling resampling:
* Distribution of the target feature:
<img src="https://github.com/ltadrummond/challenge-model-comparisson/blob/main/visuals/target_features_distribution.png">


* Comparisson of the performance of different algorythms:




# Conclusion
To achieve a better performing model, normalizing and resampling were two important preprocessing steps. Also being able to compare classifiers and tunning the hyperparameters, considering overfitting, are needed to be taken into consideration.
The best performed model was Random Forest with an accuracy of......
Accuracy was the chosen metric in this case, since it was possible to evaluate the model on a balanced dataset.



# Future prospects


# Time-line
30-11-2021 to 02-12-2021

# Contributors
| Name                  | Github                                 |
|-----------------------|----------------------------------------|
|Leonor Drummonnd      | https://github.com/ltadrummond              |
