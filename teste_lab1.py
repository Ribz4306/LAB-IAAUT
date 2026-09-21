import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_pickle('Xtrain.pkl')

print(df_train.shape)      # (71, 1) -> 71 sequências
print(df_train.columns)    # ['Skeleton_Sequence']
print(df_train.head())     # mostra só o início de cada array

X = np.concatenate(df_train['Skeleton_Sequence'].to_numpy())
print(X.shape)