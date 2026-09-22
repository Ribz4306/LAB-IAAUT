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

    # distância eucladiana de cada pose ao seu centroide: (N,)
    dist2 = np.sum((X - centr_point)**2, axis=1)

    # custo total: um único número
    return dist2

# TODO: calcule a fun¸c~ao de custo
    return

def kmeans(X, K, flag):
    #Aplica o algoritm k-means `a matriz de dados X.

    #TODO: Implemente o algoritmo do k-means, com um crit´erio de paragem apropriado.
    #Use as fun¸c~oes definidas acima

    total_cost= kmeans_custo(X_norm,C, s).sum()
    return

C = kminit(X_norm, 3, 0)
s = np.random.randint(3, size=X_norm.shape[0])   # atribuições ao calhas
print(kmeans_custo(X_norm, C, s))