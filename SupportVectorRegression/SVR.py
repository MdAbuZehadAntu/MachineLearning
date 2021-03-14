import pandas as pd
import numpy as np

data=pd.read_csv("Position_Salaries.csv")
X=data.to_numpy()[:,1:-1]
y=data.to_numpy()[:,-1:]

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X=sc_x.fit_transform(X)
y=sc_y.fit_transform(y)

from sklearn.svm import SVR
svr=SVR(kernel="rbf")
svr.fit(X,y)
print(sc_y.inverse_transform(svr.predict(sc_x.transform([[6.5]]))))

import matplotlib.pyplot as plt
plt.scatter(X,y,color="Blue")
plt.plot(X,svr.predict(X),color="red")
plt.show()

X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color="Blue")
plt.plot(X_grid,svr.predict(X_grid),color="red")
plt.show(
    
)
