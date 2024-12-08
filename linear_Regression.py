import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#importing datasets
dataset = pd.read_csv("BARUN.csv")#importing dataset
#print(dataset.isnull().sum())
#droping null values
dataset = dataset.dropna()
#print(dataset.isnull().sum())
X = dataset.iloc[:, list(range(2,5)) + list(range(7,9))].values
Y = dataset.iloc[:, 5].values

# conforming if independent variable data is null or not null
# print(dataset['open'].notnull())
# print(dataset['high'].notnull())
# print(dataset['low'].notnull())
# print(dataset['traded_quantity'].notnull())
# print(dataset['traded_amount'].notnull())
# # print(dataset)
# # Exploring the data wih pandas 
# pd.set_option('display.float_format', '{:.4f}'.format)
# pd.set_option('display.max_columns',9)
# print(pd.set_option('display.width',None))
# print(X)
#splitting the dataset into training and test 
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2, random_state = 1)
#print(X_train)
#print(y_train)
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)
# print("Predictions:", y_pred)
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
print("R² Score for linear regression is:  ", r_squared)
"""
from sklearn.model_selection import cross_val_score
scores = cross_val_score(regressor, X, Y, cv=5)
print("Cross-validation scores:", scores)
print("Average score:", np.mean(scores))
from sklearn.linear_model import Ridge

# Initialize Ridge Regression with a regularization parameter (alpha)
ridge_regressor = Ridge(alpha=1.0)

# Train the model
ridge_regressor.fit(X_train, y_train)

# Predict using the trained model
y_pred = ridge_regressor.predict(X_test)

# Evaluate the model
r_squared = r2_score(y_test, y_pred)
print("R² Score:", r_squared)

# Cross-validation
scores = cross_val_score(ridge_regressor, X, Y, cv=5)
print("Cross-validation scores:", scores)
print("Average score:", np.mean(scores))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Import dataset
dataset = pd.read_csv("BARUN.csv")
dataset = dataset.dropna()  # Drop rows with missing values

# Feature Engineering: Add previous closing prices as lag features
# Adding lagged features for previous 1 to 10 days (adjust as needed)
for lag in range(1, 11):
    dataset[f'close_lag_{lag}'] = dataset['close'].shift(lag)

# Drop rows with NaN values after creating lag features (first 10 rows will be NaN)
dataset = dataset.dropna()

# Selecting features (lagged features) and target variable (close price 10 days later)
X = dataset[['close_lag_' + str(i) for i in range(1, 11)]].values  # Use previous 10 days' closing prices
Y = dataset['close'].shift(-10).dropna().values  # Target is the close price after 10 days

# Ensure X and Y have the same length
X = X[:-10]  # Drop the last 10 rows of X to match the length of Y

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize Ridge Regression model with regularization parameter (alpha)
ridge_regressor = Ridge(alpha=1.0)

# Train the model on the training data
ridge_regressor.fit(X_train, y_train)

# Predict the price after 10 days using the trained model
y_pred = ridge_regressor.predict(X_test)

# Evaluate the model
r_squared = r2_score(y_test, y_pred)
print("R² Score:", r_squared)

# Cross-validation
scores = cross_val_score(ridge_regressor, X, Y, cv=5)
print("Cross-validation scores:", scores)
print("Average cross-validation score:", np.mean(scores))
# Step 1: Prepare the dataset without lagged features
# Assuming you have a dataset with columns like 'open', 'high', 'low', 'traded_quantity', 'traded_amount'
# and 'close' (current price)

# For predicting the close value after 10 days, we shift the target column:
dataset['target'] = dataset['close'].shift(-10)

# Drop the last 10 rows because they won't have a target value after 10 days
dataset = dataset.dropna()

# Step 2: Select the features (without lagged ones)
features = ['open', 'high', 'low', 'traded_quantity', 'traded_amount']

# Step 3: Separate features and target
X = dataset[features]
y = dataset['target']

# Step 4: Standardize the features if needed
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train the model (e.g., Linear Regression)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_scaled, y)

# Step 6: Make predictions for the next 10 days
predictions = model.predict(X_scaled)

# Example: Predict share value after 10 days
print("Predicted share price after 10 days: ", predictions[-1])



"""