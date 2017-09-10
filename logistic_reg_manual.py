import pandas as pd
import numpy as np

df = pd.read_csv("output.csv")

X = df[df.columns[0]]
X = np.array(X)

Y = df[df.columns[1]]
Y = np.array(Y)

