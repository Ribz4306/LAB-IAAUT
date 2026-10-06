import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# NAO MODIFICAR A SEED
# flag = 0 --> random
# flag = 1 --> km++
np.random.seed(555)

df_train = pd.read_pickle('Xtrain.pkl')

X = np.concatenate(df_train['Skeleton_Sequence'].to_numpy())

#Normalizaçao dos dados - centrar + escalamento (recomendando e crıtico para o PCA)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

def kminit(X, K, flag):
    N = X.shape[0]
    C = []
    init_centr_idx= np.random.randint(N)
    C.append(X[init_centr_idx])    
    if flag == 0:
        idx = np.random.choice(N, K - 1, replace=False)
        while init_centr_idx in idx: # caso raro --> repetiu o primeiro
            idx = np.random.choice(N, K - 1, replace=False)
        C.extend(X[idx])

    else: # k-means++
        for _ in range(K - 1):
            C_atual = np.array(C)                  
            M = np.zeros((N, len(C_atual)))

            # distância ao quadrado de cada pose a cada centroide já escolhido
            for k in range(len(C_atual)):
                M[:, k] = np.sum((X - C_atual[k])**2, axis=1)

            # distância de cada pose ao centroide MAIS PRÓXIMO: (N,)
            d2 = np.min(M, axis=1)

            # probabilidade
            prob = d2 / np.sum(d2)

            new_idx = np.random.choice(N, p=prob)
            C.append(X[new_idx])
                
    return np.array(C)

def kmeans_custo(X, C, s):
    centr_point = C[s]

    # distância euclidiana de cada pose ao seu centroide: (N,)
    dist_eucl = np.sum((X - centr_point)**2, axis=1)

    total_cost= np.sum(dist_eucl)

    return total_cost

def kmeans(X, K, flag):
    # Aplica o algoritm k-means a matriz de dados X.
    C = kminit(X,K,flag)
    prev_cost = np.inf
    N = X.shape[0]

    while True:
        M = np.zeros((N, K))

        for k in range(K):
            M[:, k] = np.sum((X - C[k])**2, axis=1)     # distância ao quadrado de cada pose ao centroide k: (N,)

        s = np.argmin(M, axis=1)                        # para cada pose, o centroide mais próximo

        total_cost = kmeans_custo(X, C, s)

        # Comparacao dos custos
        if prev_cost - total_cost < 1e-6:
            break
        prev_cost = total_cost

        # Atualizacao dos centroides
        for k in range(K):
            cluster = X[s == k]
            if len(cluster) > 0:
                C[k] = np.mean(cluster, axis=0)

    return C, s