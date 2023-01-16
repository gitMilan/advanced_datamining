import numpy as np

class Perceptron():

    def __init__(self, dim):
        self.dim = dim
        self.bias = 0
        self.weights = [0.0]*dim



    def predict(self, xs):
        """Implements the predict formula"""
        linear_output = np.dot(xs, self.weights) + self.bias

        # print(np.where(linear_output >= 0, 1, -1))
        a = np.where(linear_output >= 0, 1.0, -1.0)

        return a.tolist()

    def partial_fit(self, xs, ys):
        """Partial fit function which updates the weights and bias of the model"""

        for x, y in zip(xs, ys):
            y_p = self.predict(x)
            self.bias = self.bias - (y_p - y)
            for i, w in enumerate(self.weights):
                self.weights[i] = self.weights[i] - (y_p - y)*x[i]


    def fit(self, xs, ys, *, epochs=0):
        """Fit function, if no user set epochs is given it wil continue using the
            partial fit method to update the weights and bias until all instances are correctly predicted"""
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
        self.bias = 0
        self.weights = [0.0]*dim



    def predict(self, xs):
        """Implements the predict formula"""
        linear_output = np.dot(xs, self.weights) + self.bias


        return linear_output.tolist()

    def partial_fit(self, xs, ys, *, alpha=0.01):
        """Partial fit function which updates the weights and bias of the model"""

        for x, y in zip(xs, ys):
            y_p = self.predict(x)
            self.bias = self.bias - alpha*(y_p - y)
            for i, w in enumerate(self.weights):
                self.weights[i] = self.weights[i] - alpha*(y_p - y)*x[i]


    def fit(self, xs, ys, *, alpha=0.01, epochs=100):
        """Fit function"""
        if epochs:
            for _ in range(epochs):
                self.partial_fit(xs, ys, alpha=alpha)



    def __repr__(self):
        text = f'LinearRegression(dim={self.dim})'
        return text