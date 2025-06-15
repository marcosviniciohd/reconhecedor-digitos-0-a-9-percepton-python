import numpy as np

# Padrões de entrada para os dígitos 0 a 9 (5x3 = 15 + bias = 16)
# Cada linha representa um dígito (0 a 9)
X = np.array([
    [1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,1], # 0
    [1,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1], # 1
    [1,1,1,0,0,1,1,1,1,1,1,0,1,1,1,1], # 2
    [1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,1], # 3
    [1,0,1,1,1,1,1,1,1,0,0,1,0,0,1,1], # 4
    [1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1], # 5
    [1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1], # 6
    [1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,1], # 7
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], # 8
    [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1], # 9
])

# Saída esperada para cada perceptron (one-hot)
T = np.eye(10) * 2 - 1  # 1 para o dígito correto, -1 para os demais

class PerceptronDigitos:
    def __init__(self, n_classes=10, n_features=16):
        self.n_classes = n_classes
        self.n_features = n_features
        self.w = np.zeros((n_classes, n_features))
        self.epocas = 0

    def saida(self, yent, limiar):
        if yent > limiar:
            return 1
        elif yent < -limiar:
            return -1
        else:
            return 0

    def fit(self, X, T, alfa=0.1, limiar=0.1):
        mudou = True
        self.epocas = 0
        self.w = np.zeros((self.n_classes, self.n_features))
        while mudou:
            mudou = False
            for i in range(X.shape[0]):
                f = np.zeros(self.n_classes)
                for k in range(self.n_classes):
                    yent = np.dot(X[i], self.w[k])
                    f[k] = self.saida(yent, limiar)
                if not np.array_equal(f, T[i]):
                    for k in range(self.n_classes):
                        self.w[k] += alfa * (T[i][k] - f[k]) * X[i]
                    mudou = True
            self.epocas += 1

    def predict(self, x, limiar=0.1):
        yents = [np.dot(x, self.w[k]) for k in range(self.n_classes)]
        return np.argmax(yents)

# Teste do algoritmo
if __name__ == "__main__":
    perceptron = PerceptronDigitos()
    perceptron.fit(X, T, alfa=0.1, limiar=0.1)
    print(f"Épocas de treinamento: {perceptron.epocas}")
    acertos = 0
    for i in range(10):
        pred = perceptron.predict(X[i], limiar=0.1)
        print(f"Esperado: {i}, Predito: {pred}")
        if pred == i:
            acertos += 1
    print(f"Acurácia: {acertos}/10")
