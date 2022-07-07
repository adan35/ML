# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# training the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set
y_pred = regressor.predict(X_test)

# visualizing the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary $')
plt.show()

# visualizing the test set result
plt.scatter(X_test, y_test, color='orange')
plt.plot(X_train, regressor.predict(X_train), color='yellow')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary $')
plt.show()
