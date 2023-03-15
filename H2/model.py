import numpy as np
import math
import random as rnd


def linear(a):
    return a


def sign(x):
 return 1/(1 + np.exp(-x))


def tanh(a):
    return math.tanh(a)

def hinge(yhat, y):
    return np.sum(max(0 , 1 - (yhat * y)))


def mean_squared_error(yhat, y):
    return (yhat-y)**2

def mean_absolute_error(yhat, y):
    return abs(yhat-y)


def derivative(function, delta=0.01):
    def wrapper_derivative(x, *args):
        return (function(x+delta, *args) - function(x-delta, *args)) / (2*delta)

    wrapper_derivative.__name__ = function.__name__ + ','
    wrapper_derivative.__qualname__ = function.__qualname__ + ','
    return wrapper_derivative


class Neuron():

    def __init__(self, dim, activation=linear, loss=mean_absolute_error):
        self.dim = dim
        self.bias = 0.0
        self.weights = [1.0] * dim
        self.activation = activation
        self.loss=loss

    def predict(self, xs):

        predicts = []
        for instance in xs:

            y = self.activation(self.bias + np.dot(self.weights, instance))
            predicts.append(y)

        return predicts

    def predict_single_instance(self, instance):
        return self.activation(self.bias + np.dot(self.weights, instance))


    def partial_fit(self, xs, ys, *, alpha=0.01):

        for x ,y in zip(xs, ys):
            yhat = self.activation(self.bias + np.dot(self.weights, x))



            activation_derivative = derivative(self.activation)
            activation = activation_derivative(yhat)


            loss_derivative = derivative(self.loss)
            loss = loss_derivative(yhat, y)

            self.bias = self.bias - alpha * loss * activation
            for i, w in enumerate(self.weights):
                lel = self.weights[i] - alpha * loss * activation * x[i]
                self.weights[i] = lel



    def fit(self, xs, ys, * , alpha=0.01, epochs=200):
        for _ in range(epochs):
            self.partial_fit(xs,ys, alpha=alpha)


    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text