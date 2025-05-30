# Multiple regression 

#importing the liabaries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets 

dataset = pd.read_csv('multiple.csv')
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , :-1].values

#Encoding categories 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='pass through')
X =np.array(ct.fit_transform(X))
print(X)

#splitting the data into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#training multiple linear regression model in training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_test,Y_train)

#predicting the test result 
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1)),Y_test.reshape(len(Y_test),1),1))
