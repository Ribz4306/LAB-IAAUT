import numpy as np

class ArvoreDecisao:
    class Node:
        def __init__(self, feature=None, threshold=None, esquerda=None, direita=None, *, valor=None):
            self.feature = feature
            self.limiar = threshold
            self.esquerda = esquerda
            self.direita = direita
            self.valor = valor

        def is_leaf_node(self):
            return self.valor is not None

        @property                      
        def value(self):             # alias de self.valor, necessário para a imprimir_arvore (usa no.value)
            return self.valor 

    def __init__(self, profundidade_maxima=None):
        self.profundidade_maxima = profundidade_maxima
        self.raiz = None

    def _entropy(self, y):
        _, contagens = np.unique(y, return_counts=True)
        p = contagens / len(y)
        
        return -np.sum(p * np.log2(p))              # i(m) = -Σ P(k|m) log2 P(k|m)

    def _mutual_information(self, y, y_esquerda, y_direita):
    #TO DO - CALCULO DA INFORMAÇÃO MUTUA
        deltaI= self._entropy(y) - ((len(y_esquerda)/len(y))*self._entropy(y_esquerda) 
                                    + len(y_direita)/len(y)*self._entropy(y_direita))
        return deltaI

    def _best_split(self, X, y):
        melhor_ganho = -1
        melhor_feature = None
        melhor_limiar = None

        n_features = X.shape[1]               

        # Ciclo exterior: cada feature, pela ordem das colunas
        for j in range(n_features):
            coluna = X[:, j]

            valores = np.unique(coluna)
            limiares = (valores[:-1] + valores[1:]) / 2

            # Ciclo interior: cada limiar
            for t in limiares:
                var = coluna <= t           # vetor com True e False que indica o que deve ser guardado no vetor y ou não
                y_esquerda = y[var]
                y_direita = y[~var]

                ganho = self._mutual_information(y, y_esquerda, y_direita)

                if ganho > melhor_ganho:
                    melhor_ganho = ganho
                    melhor_feature = j
                    melhor_limiar = t

        return melhor_feature, melhor_limiar, melhor_ganho

    def _votacao(self, y):
        valores, contagens = np.unique(y, return_counts=True)

        empatadas = valores[contagens == contagens.max()]
        
        return empatadas.max() # em empate, a classe de maior valor

    def _build_tree(self, X, y, profundidade=0):
        # Paragem por profundidade máxima
        if self.profundidade_maxima is not None and profundidade >= self.profundidade_maxima:
            return self.Node(valor=self._votacao(y))

        feature, limiar, ganho = self._best_split(X, y)

        # Paragem se não há ganho positivo (nó puro, ou nenhuma divisão ajuda)
        if ganho <= 0:
            return self.Node(valor=self._votacao(y))

        # Dividir os exemplos pela pergunta "X[:, feature] <= limiar"
        var = X[:, feature] <= limiar

        esquerda = self._build_tree(X[var], y[var], profundidade + 1)
        direita  = self._build_tree(X[~var], y[~var], profundidade + 1)

        return self.Node(feature, limiar, esquerda, direita)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.raiz = self._build_tree(X, y)   # profundidade começa em 0 (valor por defeito)
        return self

    def predict(self, X):
        X = np.asarray(X)
        previsoes = []

        for x in X:                              
            no = self.raiz                          
            while not no.is_leaf_node():            # Enquanto houver pergunta para responder
                if x[no.feature] <= no.limiar:      #  x_j <= s ?
                    no = no.esquerda                
                else:
                    no = no.direita                 
            previsoes.append(no.valor)              # Guarda a classe

        return np.array(previsoes)