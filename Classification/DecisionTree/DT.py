import numpy as np
import pandas as pd

data=pd.read_csv("Position_Salaries.csv")

X=data.to_numpy()[:,1:-1]
y=data.to_numpy()[:,-1]
print(X.shape)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=0)
dt.fit(X,y)
print(dt.predict([[6.5]]))

import matplotlib.pyplot as plt

X_grid=np.arange(min(X),max(X),0.001)
X_grid=X_grid.reshape(X_grid.shape[0],1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,dt.predict(X_grid),color="green")
plt.show()