import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# NAO MODIFICAR A SEED
# flag = 0 --> random
# flag = 1 --> km++
np.random.seed(555)

df_train = pd.read_pickle('Xtrain.pkl')

# todas as poses numa só matriz (8374, 66)
X = np.concatenate(df_train['Skeleton_Sequence'].to_numpy())

# normalização
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

def kminit(X, K, flag):
    N = X.shape[0]
    C = []
    first_idx = np.random.randint(N)
    C.append(X[first_idx])

    if flag == 0:
        idx = np.random.choice(N, K - 1, replace=False)
        while first_idx in idx:  # caso raro --> repetiu o primeiro
            idx = np.random.choice(N, K - 1, replace=False)
        C.extend(X[idx])

    else:  # k-means++
        for _ in range(K - 1):
            C_current = np.array(C)
            M = np.zeros((N, len(C_current)))

            # distância ao quadrado de cada pose a cada centroide já escolhido
            for k in range(len(C_current)):
                M[:, k] = np.sum((X - C_current[k])**2, axis=1)

            # distância de cada pose ao centroide MAIS PRÓXIMO: (N,)
            min_sq_dist = np.min(M, axis=1)

            # probabilidade
            prob = min_sq_dist / np.sum(min_sq_dist)

            new_idx = np.random.choice(N, p=prob)
            C.append(X[new_idx])

    return np.array(C)


def kmeans_custo(X, C, s):
    assigned_centroids = C[s]

    # distância euclidiana de cada pose ao seu centroide: (N,)
    sq_dist = np.sum((X - assigned_centroids)**2, axis=1)

    total_cost = np.sum(sq_dist)

    return total_cost


def kmeans(X, K, flag):
    # Aplica o algoritm k-means a matriz de dados X.
    C = kminit(X, K, flag)
    prev_cost = np.inf
    N = X.shape[0]

    while True:
        M = np.zeros((N, K))

        for k in range(K):
            M[:, k] = np.sum((X - C[k])**2, axis=1)  # distância ao quadrado de cada pose ao centroide k: (N,)

        s = np.argmin(M, axis=1)  # para cada pose, o centroide mais próximo

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