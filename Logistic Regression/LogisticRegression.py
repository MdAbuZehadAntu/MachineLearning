import pandas as pd
import numpy as np

data = pd.read_csv("Social_Network_Ads.csv")

X = data.loc[:,["Age","EstimatedSalary"]]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_X.fit(X_train)
X_train=sc_X.transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix

print("Accuracy score",accuracy_score(y_pred,y_test)*100,"%")

cm=confusion_matrix(y_true=y_test,y_pred=y_pred)
print(cm)

