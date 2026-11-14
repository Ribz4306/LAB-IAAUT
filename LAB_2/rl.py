import numpy as np
#ACRESCENTAR BIBLIOTECAS QUE SEJAM NECESSÁRIAS

class MulticlassLogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, num_epocas=100):
        self.lr = learning_rate
        self.epocas = num_epocas
        self.pesos = None
        self.offset = None
        self.loss_history = []

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def fit(self, X, y, num_classes):
        N , p = X.shape
        self.pesos= np.zeros((p,num_classes))
        self.offset = np.zeros(num_classes)
        self.loss_history = []
        Y = np.eye(num_classes)[y]

        for epoca in range(self.epocas):
            for i in range(N):
                x = X[i]                          # features da amostra i, forma (p,)
                y_i = Y[i]                        # one-hot da amostra i, forma (K,)

                z = self.pesos.T @ x + self.offset    #  z = Wᵀx + b
                prob = self._softmax(z)               
                erro = prob - y_i                     

                self.pesos = self.pesos - self.lr * np.outer(x, erro)   # W <- W − η·x·erroT
                self.offset = self.offset - self.lr * erro              # b <- b − η·erro

            Z = X @ self.pesos + self.offset
            P = self._softmax(Z)
            custo = -np.sum(Y * np.log(P + 1e-15))      # 1e-15 para evitar o log(0)
            self.loss_history.append(custo)


    def predict(self, X):
        Z = X @ self.pesos + self.offset
        return np.argmax(Z, axis=1) 

if __name__ == "__main__":
    # 1) Exemplo numérico feito à mão
    m = MulticlassLogisticRegressionSGD(learning_rate=0.1, num_epocas=1)
    m.fit(np.array([[2, 1]]), np.array([1]), num_classes=3)
    print(m.pesos)
    print(m.offset)
    print(m.loss_history)

    # 2) Dados de brincar: 3 grupos bem separados
    X = np.array([[0, 0], [0.5, 0.2],      # classe 0: perto da origem
                  [5, 0], [5.5, 0.3],      # classe 1: x₁ grande
                  [0, 5], [0.2, 5.5]])     # classe 2: x₂ grande
    y = np.array([0, 0, 1, 1, 2, 2])
    m = MulticlassLogisticRegressionSGD(learning_rate=0.1, num_epocas=50)
    m.fit(X, y, num_classes=3)
    print(m.loss_history[:3], m.loss_history[-1])
    print(m.predict(X))
    print(m.predict(np.array([[4.8, 0.1], [0.1, 4.9], [0.3, 0.1]])))     