#importing  the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#importing the datasets
dataset = pd.read_csv('Position_salaries.csv')
X=dataset.iloc[:, 1:-1].values 
Y=dataset.iloc[:, 1:-1].values
print(X)
