import numpy as np
import pandas as pd

data = pd.read_csv("50_Startups.csv")

Xt = data.iloc[:,: -1]
X = data.iloc[:,: -1].values
y = data.iloc[:, -1].values
# print(type(X))
# quit()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder="passthrough")
X = ct.fit_transform(X)

X = X[:, 1:]  # avoiding dummy variable trap

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(y_pred.shape[0],1),y_test.reshape(y_test.shape[0],1)),axis=1))


import statsmodels.api as sm

X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
index = list()
for i in range(X.shape[1]):
    index.append(i)
X_opt = np.array(X[:, index], dtype=float)
# print(X_opt)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# print(regressor_OLS.summary())
# quit()
index.remove(2)
# print(index)

X_opt = np.array(X[:, index], dtype=float)
# print(X_opt)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# print(regressor_OLS.summary())
index.remove(1)

X_opt = np.array(X[:, index], dtype=float)
# print(X_opt)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# print(regressor_OLS.summary())
index.remove(4)
X_opt = np.array(X[:, index], dtype=float)
# print(X_opt)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# print(regressor_OLS.summary())
index.remove(5)

X_opt = np.array(X[:, index], dtype=float)
# print(X_opt)

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# print(regressor_OLS.summary())
# index.remove(2)




X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
np.set_printoptions(precision=2)
print()
print()
print()
print(np.concatenate((y_pred.reshape(y_pred.shape[0],1),y_test.reshape(y_test.shape[0],1)),axis=1))


