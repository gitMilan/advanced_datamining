
import numpy as np




def sign(a):
    if a > 0:
        return 1
    return -1





class Perceptron():

    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0]*dim


    def predict(self, xs):
        """Takes a list of instances and predicts the outcome, returns a list of predictions"""
        predictions = []
        for instance in xs:
            activation = sign(self.bias + np.dot(instance, self.weights))
            predictions.append(activation)
        return predictions


    def partial_fit(self, xs, ys):
        for x, y in zip(xs, ys):
            predict = sign(self.bias + np.dot(x, self.weights))
            self.bias = self.bias - (predict - y)
            for i, w in enumerate(self.weights):
                self.weights[i] = self.weights[i] - (predict - y)*x[i]


    def fit(self, xs, ys, *, epochs=0):
        if epochs:
            for _ in range(epochs):
                self.partial_fit(xs, ys)
        else:
            while self.predict(xs) != ys:
                self.partial_fit(xs, ys)



    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text


class LinearRegression():

    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0]*dim


    def predict(self, xs):
        """Takes a list of instances and predicts the outcome, returns a list of predictions"""
        predictions = []
        for instance in xs:
            activation = sign(self.bias + np.dot(instance, self.weights))
            predictions.append(activation)
        return predictions


    def partial_fit(self, xs, ys, *, alpha=0.01):
        for x, y in zip(xs, ys):
            predict = sign(self.bias + np.dot(x, self.weights))
            self.bias = self.bias - alpha*(predict - y)
            for i, w in enumerate(self.weights):
                self.weights[i] = self.weights[i] - alpha*(predict - y)*x[i]


    def fit(self, xs, ys, *, alpha=0.01, epochs=100):
        """Fit function"""
        if epochs:
            for _ in range(epochs):
                self.partial_fit(xs, ys, alpha=alpha)



    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text