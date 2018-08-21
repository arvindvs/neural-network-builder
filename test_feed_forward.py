import numpy as np
import time

from src.neural_network import NeuralNetwork

layers = [2, 10, 10, 5]

network = NeuralNetwork(layers)

print(network.feed_forward([10, 10]))
