import numpy as np
import time

class Neuron:
    def _init_(self, prevLayerSize):
        self.value = 0
        self.bias = np.random.rand()
        self.error = 0
