import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset 

dataset= pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[: ,1:-1].values
Y = dataset.iloc[: ,-1].values
#print(X)
#print(Y)

# Training in random forest regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)
print(regressor.fit(X,Y))
#predictiong using  random forest regressor

print('the predicted value using the random forest regression is :\t',regressor.predict([[6.5]]))
#print(regressor.predict([[5.5]]))

#now visualizing the given data 
x_grid = np.arange(X.min(),X.max(),0.1)

x_grid = x_grid.reshape(-1, 1) 
plt.scatter(X,Y,color='orange')
plt.plot(x_grid, regressor.predict(x_grid), color ='blue')
#plt.plot(X,regressor.predict(x_grid))
plt.title('Truth or Bluff(decision tree regression)')
plt.xlabel('level')
plt.ylabel('Salary')

plt.show()