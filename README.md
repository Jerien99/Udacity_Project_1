# Political Conflict Prediction in Africa and the Middle East
This is the repository with all necessary documents for the first project of the Data Scientist nanodegree by Udacity

## Libraries Used

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
```
Make sure that NumPy version 1.24.0 is installed for compatibility with the SHAP-package. The notebook does contain a code block that installs this specific NumPy version.

## Project Motivation
This project is part of the Udacity Data Scientist nanodegree. Since machine learning is utilized more and more in political science contexts, it is interesting to investigate the performance of these methods when predicting geopolitical phenomena like political instability or intrastate violence. This project marries political science with data science by employing a Random Forest Model for conflict prediction in Africa and the Middle East.

## Files
This repository contains two files, next to this ReadMe. Firstly, the Data_World_Bank.csv contains all data from the World Bank database, which is available through https://databank.worldbank.org/. Secondly, the analysis.ipynb contains the entire analysis executed for this project, which includes data pre-processing, data exploration, fitting the Random Forest Model and evaluating its performance.

## Summary of Analysis Results
This project aims to predict political instability in Africa and the Middle East using various features from the World Bank Database, such as merchandise trade, forested area, armed personnel, GDP growth and inflation. 68 African and Middle Eastern countries were analyzed, with data from 2000 until 2023. Political instability was operationalized as a country in a given year having a Political Stability and Absence of Conflict-score below -1, which is at least one standard deviation away from the global mean. Because of its robustness to class imbalance and a wide variety of input features, a Random Forest Model was chosen to perform the prediction. 

The model is evaluated using accuracy, precision, recall and F1-score, of which the latter two are the most important due to false negatives being an especially expensive mistake in this context. With an accuracy of 0.82, precision of 0.83, recall of 0.52 and F1-score of 0.64, the model is quite weak and is not able to predict political instability very well. The low recall score shows that false negatives are a problem for the model. And since not predicting instability when there it is there is a particularly costly mistake, this model is suboptimal. This shows the complexity of political instability and the difficulty of operationalizing this phenomenon effectively in a modelling setting.

SHAP-values show that merchandise trade and forested area are the most important features in the prediction, with both features showing a positive correlation with political instability. This is actually in accordance with prior research done by Muchlinski et al. (2016), which is available via https://www.jstor.org/stable/24573207. 

## Acknowledgements
A sincere thanks to the World Bank for amassing a rich and easy-to-use database of political and economic data on every nation worldwide, and thanks to all lecturers in the second course of the Udacity Data Scientist nanodegree for providing me with the technical insights necessary to complete this project.