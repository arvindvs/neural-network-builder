import numpy as np
import argparse
import time

# Classes
from src.neuron import Neuron

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
        softmax:
            Expects array as input
            Returns output of softmax function on input array

    Attributes:
        inputSize: number of inputs expected
        outputSize: number of possible classifications
        neurons: list of arrays representing layers, each containing Neurons
        layerWeights: list of matrices representing weights between layers
        activationFunction: choice between Sigmoid, ReLU, and Tanh
    """
    def __init__(self, layerSizes, activationFunction='Sigmoid'):
        self.inputSize = layerSizes[0]

        self.outputSize = layerSizes[-1]

        self.neurons = []
        for i in range(len(layerSizes) - 1):
            temp = []
            for j in range(layerSizes[i+1]):
                temp.append(Neuron(layerSizes[i]))
            self.neurons.append(temp)

        self.layerWeights = []
        for index in range(len(layerSizes) - 1):
            self.layerWeights.append(np.random.rand(layerSizes[index], layerSizes[index+1]))

        self.activationFunction = activationFunction

    def feed_forward(self, input):
        assert len(input) == self.inputSize
        assert range(len(self.layerWeights)) == range(len(self.neurons))

        values = input
        for weightIndex in range(len(self.layerWeights)):
            afterMatMul = np.matmul(values, self.layerWeights[weightIndex])

            afterBiasActivation = []
            assert len(afterMatMul) == len(self.neurons[weightIndex])
            for neuronIndex in range(len(afterMatMul)):
                afterBiasActivation.append(self.activate(afterMatMul[neuronIndex] + self.neurons[weightIndex][neuronIndex].bias))

            values = afterBiasActivation

        return self.softmax(values)



    def activate(self, value):
        if(self.activationFunction == 'Sigmoid'):
            return (1.0/(1.0 + np.exp(-value)))

        if(self.activationFunction == 'ReLU'):
            return value*(value > 0)

        if(self.activationFunction == 'Tanh'):
            return numpy.tanh(value)

    def softmax(self, input):
        exp_value = np.exp(input - np.max(input))
        exp_adjusted = exp_value / np.sum(exp_value)
        return exp_adjusted
