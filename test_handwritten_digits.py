import numpy as np
import pickle

from src.neural_network import NeuralNetwork

layers = [64, 20, 10]

network = NeuralNetwork(layers, 0.007)
print(network.activationFunctions)
print(network.trainMethod)
print(network.inputSize)
print(network.outputSize)

train_inputs = []
train_outputs = []

with open('test_files/handwritten_digits/optdigits_train.txt', 'r') as f:
    for line in f:
        array = list(map(int, line.split(",")))
        value = array[-1]
        output = np.zeros(10)
        assert value < 10
        output[value] = 1.0
        array = array[:-1]
        train_inputs.append(array)
        train_outputs.append(output)

network.train(train_inputs, train_outputs, 600)

pickle.dump(network, open("trained_networks/handwritten_digit_classifier.p", "wb"))

test_inputs = []
test_outputs = []

accurate_counter = 0.0
total_counter = 0.0

with open('test_files/handwritten_digits/optdigits_test.txt', 'r') as g:
    temp_array = np.array([])
    for line in g:
        array = np.array(list(map(int, line.split(","))))
        assert np.array_equal(array, temp_array) is False
        actualValue = array[-1]
        assert actualValue < 10

        array = array[:-1]
        output = network.feed_forward(array)

        predictedValue = np.argmax(output)

        if predictedValue == actualValue:
            accurate_counter = accurate_counter + 1

        total_counter = total_counter + 1

        temp_array = array

probability = accurate_counter/total_counter

print("Your accuracy is ", probability)
