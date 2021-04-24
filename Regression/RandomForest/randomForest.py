import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Position_Salaries.csv")
X=data.to_numpy()[:,1:-1]
y=data.to_numpy()[:,-1]

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=300,random_state=0)
rf.fit(X,y)

print(rf.predict([[6.5]]))

X_grid=np.arange(min(X),max(X),0.001)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,rf.predict(X_grid),color="blue")
plt.show()