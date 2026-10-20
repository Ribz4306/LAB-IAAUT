import numpy as np
import pandas as pd

df = pd.read_pickle('Xsup.pkl')
print(df.columns.tolist())
print(df.head())

def coef_estimation(X,Y):
    XtX     = X.T @ X
    XtX_inv = np.linalg.inv(XtX)     # Inversa
    XtY     = X.T @ Y
    B       = XtX_inv @ XtY          # B = (X^T X)^-1 X^T Y
  
    return B

filter_patient_4 = (df['Patient_ID'] == 4) & (df['Exercise_Id'] == 'E5')
a = np.sum(filter_patient_4)
print(a)
seqs = df.loc[filter_patient_4, 'Skeleton_Sequence'].to_list()

n_test = int(round(0.2 * len(seqs)))
train_seqs = seqs[:-n_test]

print('Total:', len(seqs), '| Treino:', len(train_seqs))

def build_matrix(sequences):
    X_list, Y_list = [], []
    for s in sequences:
        X_list.append(s[:-1])   # poses 0 .. T-2 (anterior)
        Y_list.append(s[1:])    # poses 1 .. T-1 (seguinte)

    X = np.vstack(X_list)                          # (N, 66)
    Y = np.vstack(Y_list)                          # (N, 66)
    X = np.hstack([np.ones((X.shape[0], 1)), X])   # (N, 67): coluna de 1s
    return X, Y

B = coef_estimation(X,Y)
beta0 = B[0, :]    
beta  = B[1:, :]