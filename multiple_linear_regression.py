#importing dataset from scikit learn as it provides small to big datas  of real world or made
from sklearn.datasets import fetch_california_housing
import pandas as pd

#inserting the dataset fetch_california_housing into variable california
california = pd.read_csv('cal_housing')
print(california.DESCR)
