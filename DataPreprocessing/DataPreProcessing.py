import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

st = time.time()

df = pd.read_csv("Data.csv")
# print(df)
# df=df.dropna(axis=1)
# print(df)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

X[:, 1:] = imputer.fit_transform(X[:, 1:])
# print(X)
from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
X = ct.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler

standardscaler=StandardScaler(with_mean=True)
X_train=standardscaler.fit_transform(X_train)
X_test=standardscaler.fit_transform(X_test)

print(X_train)


