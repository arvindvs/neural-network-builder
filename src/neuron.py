import numpy as np
import time

class Neuron:
    def __init__(self, prevLayerSize):
        self.value = 0
        self.bias = np.random.rand()
        self.error = 0
