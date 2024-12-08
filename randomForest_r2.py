import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('BARUN.csv')
dataset = dataset.dropna()
X = dataset.iloc[:, list(range(2, 5)) + list(range(7, 9))].values
y = dataset.iloc[:, 5].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)


from sklearn.metrics import r2_score
print("R² Score for Random forest regression is: ",r2_score(y_test, y_pred))