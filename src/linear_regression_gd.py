import numpy as np

class LinearRegressionGD:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate              #how big each step is during gradient descent
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = self.predict(X)

            #COMPUTING THE GRADIENTS

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  
            db = (1 / n_samples) * np.sum(y_pred - y)

            #Update the steps

            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            loss = self.compute_loss(y, y_pred) #computing the loss at every interation
            self.loss_history.append(loss)      #storing the interation-wise loss history

    def predict(self, X):
        return np.dot(X, self.weight) + self.bias

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
