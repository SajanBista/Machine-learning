import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

#now importing dataset from csv file
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y= dataset.iloc[:,-1].values
"""print(X)
print(Y)# here Y gave 1D value but standard scaler class expects its value in 2D array 
"""
Y=Y.reshape(len(Y),1)#now it converts it into 2D array


#Feature scaling 
from sklearn.preprocessing import StandardScaler
Sc_X= StandardScaler()
X = Sc_X.fit_transform(X)
Sc_Y=StandardScaler()
Y=Sc_Y.fit_transform(Y)
""""print(X)
print(Y)
Usually standardalization transform gives value between -3 to +3 """

# Now training the SVR(support vector model) on the whole dataset 

from sklearn.svm import SVR # support vector machine
regressor = SVR(kernel='rbf') #Radial Basis Function  it is considered as on of the best kernel model among all
regressor.fit(X,Y.ravel()) # y.ravel()  will further bring to 1D


#Predicting the new result 

print(f'The predicted salary of his previous company using svr is :\t',
      Sc_Y.inverse_transform(regressor.predict(Sc_X.transform([[6.5]])).reshape(-1, 1)))#.reshape will avoid format error


#Visualizing the SVR result
X_grid = np.arange(
    Sc_X.inverse_transform(X).min(),  #extract scalar minimum
    Sc_X.inverse_transform(X).max(),  #extract scalar maximum
    0.1  #step size
)
X_grid = X_grid.reshape((len(X_grid), 1))  # reshape to 2D for transformation is because 1D lacks second dimension for feature and is ambiguous.

plt.scatter(Sc_X.inverse_transform(X), Sc_Y.inverse_transform(Y), color='red') #visualising the original data
plt.plot(
    X_grid,
    Sc_Y.inverse_transform(regressor.predict(Sc_X.transform(X_grid)).reshape(-1, 1)),
    color='blue'
)
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
