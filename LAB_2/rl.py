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

                self.pesos = self.pesos - self.lr * np.outer(x, erro)   # W ← W − η·x·erroᵀ
                self.offset = self.offset - self.lr * erro              # b ← b − η·erro


    def predict(self, X):
    #TO DO - IMPLEMENTAR A PREDIÇÃO DA REGRESSÃO LOGÍSTICA
        pass