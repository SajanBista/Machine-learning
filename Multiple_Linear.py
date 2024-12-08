
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# load dataset
dataset = pd.read_csv('BARUN.csv')
dataset = dataset.dropna()

X = dataset.iloc[:, list(range(2, 5)) + list(range(7, 9))].values  
y = dataset.iloc[:, 5].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# make predictions
y_pred = regressor.predict(X_test)

# display predictions and actual values
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1)#not displaying here it takes lot coz datasize is big

# calculate r² score
print("R² Scorefor multiple linear regression is : ", r2_score(y_test, y_pred))
