import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# N~AO MODIFICAR A SEED
# flag = 0 --> random
# flag = 1 --> km++
np.random.seed(555)

df_train = pd.read_pickle('Xtrain.pkl')

X = np.concatenate(df_train['Skeleton_Sequence'].to_numpy())

#Normaliza¸c~ao dos dados - centrar + escalamento (recomendando e cr´ıtico para o PCA)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

def kminit(X, K, flag):
#Inicializa¸c~ao - pode ser aleat´oria ou recorrendo ao k-means++
# A inicializa¸c~ao do primeiro centr´oide ´e fixa
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
        # TODO: complete o c´odigo para inicializar os K-1 centr´oides de acordo com a flag
        pass
    return np.array(C)

def kmeans_custo(X, C, s):
#Calcula a fun¸c~ao de custo do k-means
    centr_point = C[s]

    # distância euclidiana de cada pose ao seu centroide: (N,)
    dist_eucl = np.sum((X - centr_point)**2, axis=1)

    total_cost= np.sum(dist_eucl)

    return total_cost

def kmeans(X, K, flag):
    #Aplica o algoritm k-means `a matriz de dados X.
    C = kminit(X,K,flag)
    prev_cost = np.inf

    N = X.shape[0]
    K = C.shape[0]

    M = np.zeros((N, K))

    for k in range(K):
        M[:, k] = np.sum((X-C[k])**2, axis=1)   # distância ao quadrado de cada pose ao centroide k: (N,)

    s = np.argmin(M, axis=1)               # para cada pose, o centroide mais próximo

    total_cost = kmeans_custo(X,C,s)

    # Comparação dos custos    
    if prev_cost - total_cost < 1e-6:
        return C, s
    prev_cost = total_cost

    # Atualização dos centroides
    for k in range(K):
        poses_do_cluster = X[s == k]
        if len(poses_do_cluster) > 0:
            C[k] = np.mean(poses_do_cluster, axis=0)

    return C, s
