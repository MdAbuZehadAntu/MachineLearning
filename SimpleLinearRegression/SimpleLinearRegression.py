import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("Salary_Data.csv")

X=dataset.iloc[:,:-1].values
y=dataset.to_numpy()[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression

linreg=LinearRegression()
linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)

# print(y_pred.shape)
# print(y_test.shape)


diff=y_pred-y_test
diff=diff.reshape(-1,1)
print(diff)

print()
print(linreg.predict([[7.5]])[0])


# plt.scatter(X_train,y_train,color="blue")
#
# plt.plot(X_train,linreg.predict(X_train),color="red")
# plt.show()
# plt.scatter(X_test,y_test,color="black")
# plt.show()
# plt.scatter(X_test,linreg.predict(X_test),color="red")
# plt.show()


