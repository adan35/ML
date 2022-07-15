# Import Libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data

# Split the Dataset into the Training Set and Test Set

# Training the Multiple Linear Regression model on the Training set

# Predicting the Test set results