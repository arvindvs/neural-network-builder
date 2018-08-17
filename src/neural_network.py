import numpy as np
import time

class NeuralNetwork:
    def _init_(self, layerSizes, activationFunction):
        self.inputSize = layerSizes[0]
        self.outputSize = layerSizes[-1]
        self.layerWeights = {}
        for index in range(layerSizes.size - 1):
            self.layerWeights[index] = np.random.rand(layerSizes[index], layerSizes[index+1])
        self.activationFunction = activationFunction
