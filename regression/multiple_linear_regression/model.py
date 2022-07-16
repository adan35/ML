# Import Libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# Split the Dataset into the Training Set and Test Set

# Training the Multiple Linear Regression model on the Training set

# Predicting the Test set results