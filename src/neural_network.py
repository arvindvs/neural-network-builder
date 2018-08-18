import numpy as np
import argparse
import time

# Classes
from .neuron import Neuron

class NeuralNetwork:
    """
    Neural network implementation using Neuron object and network weights

    Functions:
        _init_:
            Expects array of layerSizes, type of activation functions
            Initializes network with random weights
        feed_forward:
            Expects array of input values in the shape of self.inputSize
            Performs forward propagation through network to return output array
        activate:
            Expects value to perform activation on
            Outputs activation

    Attributes:
        inputSize: number of inputs expected
        outputSize: number of possible classifications
        neurons: list of arrays representing layers, each containing Neurons
        layerWeights: list of matrices representing weights between layers
        activationFunction: choice between Sigmoid, ReLU, and Tanh
    """
    def _init_(self, layerSizes, activationFunction='Sigmoid'):
        self.inputSize = layerSizes[0]

        self.outputSize = layerSizes[-1]

        self.neurons = []
        for i in range(layerSizes.size - 1):
            temp = []
            for j in range(layerSizes[i+1]):
                temp[j] = Neuron(layerSizes[i])
            self.neurons[i] = temp

        self.layerWeights = []
        for index in range(layerSizes.size - 1):
            self.layerWeights[index] = np.random.rand(layerSizes[index], layerSizes[index+1])

        self.activationFunction = activationFunction

    def feed_forward(self, input):
        assert input.size == self.inputSize

    def activate(self, value):
        if(self.activationFunction == 'Sigmoid'):
            return (1.0/(1.0 + np.exp(-value)))

        if(self.activationFunction == 'ReLU'):
            return value*(value > 0)

        if(self.activationFunction == 'Tanh'):
            return numpy.tanh(value)
