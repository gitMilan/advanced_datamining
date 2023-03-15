from collections import Counter
from copy import deepcopy
import numpy as np
import math
from numpy import mean
from numpy.random import rand

def linear(a):
    return a

def mean_squared_error(yhat, y):
    return (yhat-y)**2


class Layer():

    classcounter = Counter()

    def __init__(self, outputs, *, name=None, next=None):
        self.size = 1
        Layer.classcounter[type(self)] += 1


        if name is None:
            name = f'{type(self).__name__}_{Layer.classcounter[type(self)]}'

        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next


    def __repr__(self):
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text


    def add(self, next):
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)
        self.size += 1


    def __add__(self, next):
        result = deepcopy(self)
        result.add(deepcopy(next))
        self.size += 1
        return result


    def __getitem__(self, index):
        if index == 0 or index == self.name:
            return self

        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'Layer indices must be integerers or strings, not {type(index).__name__}')


    def __len__(self):
        return self.size


    def set_inputs(self, inputs):
        self.inputs = inputs


class InputLayer(Layer):
    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text


    def set_inputs(self):
        return NotImplementedError()


def xavier_init(F_in, F_out):
    limit = np.sqrt(2 / float(F_in + F_out))
    W = np.random.normal(0.0, limit, size=(F_in, F_out))

    return W


class DenseLayer(Layer):

    def __init__(self, outputs, name=None):
        """The init of the dense layer"""
        super().__init__(outputs, name=name)
        self.bias = [0.0 for my_o in range(self.outputs)]
        self.weights = [[] for my_i in range(self.outputs)]

    def set_inputs(self, inputs):
        """Sets the weights and takes the sets the input like in the parent class"""
        super().set_inputs(inputs)
        spread = math.sqrt(6 / (self.inputs + self.outputs))
        self.weights = [[np.random.uniform(-spread, spread)
                         for my_i in range(self.inputs)] for my_o in range(self.outputs)]

    def __repr__(self):
        """Returns the inputs, outputs and name"""
        text = f'DenseLayer(inputs={repr(self.inputs)}, ' \
               f'outputs={repr(self.outputs)}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text


class ActivationLayer(Layer):

    def __init__(self, outputs, activation=linear, name=None):
        super().__init__(outputs, name=name)
        self.activation = activation

    def __repr__(self):
        text = f'ActivationLayer(inputs={repr(self.inputs)}, ' \
               f'outputs={repr(self.outputs)}, name={repr(self.name)}, ' \
               f'activation={self.activation})'
        if self.next is not None:
            text += ' + ' + repr(self.next)

        return text


class LossLayer(Layer):

    def __init__(self, loss=mean_squared_error, name=None):

        self.loss = loss

    def set_inputs(self, inputs):
        """Sets the weights and takes the sets the input like in the parent class"""
        super().set_inputs(inputs)

    def add(self):
        return NotImplementedError()

    def __add__(self):
        return NotImplementedError()

    def __repr__(self):
        text = f'LossLayer(inputs={repr(self.inputs)})'
        return text