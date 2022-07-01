# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Data.csv') #change it with your file
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)

# split the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 1:3] = sc.fit_transform(X_train[:, 1:3])
X_test[:, 1:3] = sc.fit_transform(X_test[:, 1:3])
# print(X_train)
# print(X_test)