import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
# print(dataset)
X=dataset.to_numpy()[:,1:-1]
y=dataset.iloc[:,-1:].values
print(y)

from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=5)
X_poly=poly_reg.fit_transform(X)
# np.set_printoptions(suppress=True)
# print(X_poly)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

plt.scatter(X,y,color="blue")
plt.plot(X,linreg.predict(X),color="red")
plt.show()

plt.scatter(X,y,color="blue")
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color="red")
plt.show()

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))[0][0])
print(linreg.predict([[6.5]])[0][0])

X_grid=np.arange(min(X),max(X),0.1)
print(type(X_grid))
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="blue")
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color="red")
plt.show()

