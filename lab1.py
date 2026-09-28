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

# RUN
Ks = list(range(1, 21))       # K de 1 a 10
n_runs = 20

# matrizes vazias: linha = K, coluna = corrida
custos_random = np.zeros((len(Ks), n_runs))
custos_pp     = np.zeros((len(Ks), n_runs))

for i, K in enumerate(Ks):
    for r in range(n_runs):
        C, s = kmeans(X_norm, K, 0)                   # random
        custos_random[i, r] = kmeans_custo(X_norm, C, s)

        C, s = kmeans(X_norm, K, 1)                   # k-means++
        custos_pp[i, r] = kmeans_custo(X_norm, C, s)

    print(f"K = {K} feito")                           # para veres o progresso

min_random  = np.min(custos_random, axis=1)    # mínimo de cada linha: (10,)
media_random = np.mean(custos_random, axis=1)
std_random   = np.std(custos_random, axis=1)

min_pp  = np.min(custos_pp, axis=1)
media_pp = np.mean(custos_pp, axis=1)
std_pp       = np.std(custos_pp, axis=1)

plt.errorbar(Ks, media_random, yerr=std_random, fmt='o-', capsize=4, label='random')
plt.errorbar(Ks, media_pp, yerr=std_pp, fmt='s-', capsize=4, label='k-means++')
plt.xlabel('Número de centroides K')
plt.ylabel('Custo médio')
plt.xticks(Ks)
plt.legend()
plt.grid(True)
plt.show()
