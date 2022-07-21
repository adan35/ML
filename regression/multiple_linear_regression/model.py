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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
result = np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
# print(result)

# Making a single prediction (for example the profit of a startup with R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
single_pred = regressor.predict([[1, 0, 0, 160000, 130000, 300000]])
# print(single_pred)

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)
# y = 42467.52 + 86.6 x Dummy Var 1 - 873 x Dummy Var 2 + 786 x Dummy Var 3 + 0.773 x R&D + 0.0329 x Administration + 0.0366 x Marketing