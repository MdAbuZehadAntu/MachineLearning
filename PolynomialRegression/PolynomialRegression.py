import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
X=dataset.to_numpy()[:,1:-1]
y=dataset.iloc[:,-1].values


