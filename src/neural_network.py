import numpy as np


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
        # neurons: list of arrays representing layers, each containing Neurons
        layerWeights: list of matrices representing weights between layers
        biases: list of array representing
        activationFunction: choice between Sigmoid, ReLU, and Tanh
        trainMethod:
        learningRate:
    """

    def __init__(self, layerSizes, learningRate, activationFunction='ReLU',
                 trainingMethod='batchGradientDescent'):

        self.inputSize = layerSizes[0]
        self.outputSize = layerSizes[-1]

        # REPLACE NEURONS #
        # self.neurons = []
        # for i in range(len(layerSizes) - 1):
        #     temp = []
        #     for j in range(layerSizes[i+1]):
        #         temp.append(Neuron(layerSizes[i]))
        #     self.neurons.append(temp)

        self.layerWeights = []
        self.biases = []
        self.activationFunctions = []
        for index in range(len(layerSizes) - 1):
            self.layerWeights.append(np.random.rand(layerSizes[index],
                                                    layerSizes[index+1]))
            self.biases.append(np.random.rand(layerSizes[index+1]))
            self.activationFunctions.append(activationFunction)
        self.activationFunctions.append('Softmax')
        self.trainMethod = trainingMethod
        self.learningRate = learningRate

    def feed_forward(self, input, trainFlag=False):
        if trainFlag:
            weightedInputs = []
            weightedInputs.append(input)

        assert len(input) == self.inputSize
        values = self.activate(input, self.activationFunctions[0])

        for weightIndex in range(len(self.layerWeights)):
            afterMatMul = np.matmul(values, self.layerWeights[weightIndex])

            assert len(afterMatMul) == len(self.biases[weightIndex])
            afterBiasAddition = np.add(afterMatMul, self.biases[weightIndex])
            if trainFlag:
                weightedInputs.append(afterBiasAddition)

            afterActivation = self.activate(afterBiasAddition,
                                            self.activationFunctions
                                            [weightIndex + 1])

            values = afterActivation

        if not trainFlag:
            return values
        else:
            return values, weightedInputs

    def train(self, inputs, outputs, epochs=500):
        print("Starting training...\n\n")
        assert len(inputs) == len(outputs)
        # dictionary = dict(zip(inputs, outputs))

        if(self.trainMethod == 'batchGradientDescent'):
            for iterations in range(epochs):
                print("========== STARTING EPOCH", iterations, " ==========")
                weightUpdates = []
                biasUpdates = []
                for indexValue in range(len(inputs)):
                    print("Input ", indexValue, "| Epoch ", iterations)
                    input = inputs[indexValue]
                    actualOutput, weightedInputs = self.feed_forward(input,
                                                                     True)
                    desiredOutput = outputs[indexValue]

                    assert len(actualOutput) == len(desiredOutput)
                    errorValues = self.backpropagate(actualOutput,
                                                     desiredOutput,
                                                     weightedInputs)
                    if len(weightUpdates) == 0 and len(biasUpdates) == 0:
                        for index in range(len(errorValues)):
                            errorInput = [errorValues[index]]
                            activatedInput = np.transpose([weightedInputs
                                                           [index]])
                            weightValue = np.matmul(activatedInput, errorInput)
                            weightUpdates.append(weightValue)
                            biasUpdates.append(errorValues[index])
                    else:
                        for index in range(len(errorValues)):
                            errorInput = [errorValues[index]]
                            activatedInput = np.transpose([weightedInputs
                                                           [index]])
                            weightValue = np.matmul(activatedInput, errorInput)
                            weightUpdates[index] = np.add(weightUpdates[index],
                                                          weightValue)
                            biasUpdates[index] = np.add(biasUpdates[index],
                                                        errorValues[index])

                for arrIndex in range(len(weightUpdates)):
                    weightUpdates[arrIndex] = np.divide(weightUpdates[arrIndex]
                                                        , len(inputs))
                    weightUpdates[arrIndex] = np.multiply(weightUpdates
                                                          [arrIndex],
                                                          self.learningRate)
                    self.layerWeights[arrIndex] = np.subtract(self.layerWeights
                                                              [arrIndex],
                                                              weightUpdates
                                                              [arrIndex])

                for biasIndex in range(len(biasUpdates)):
                    biasUpdates[biasIndex] = np.divide(biasUpdates[biasIndex],
                                                       len(inputs))
                    biasUpdates[biasIndex] = np.multiply(biasUpdates[biasIndex]
                                                         , self.learningRate)
                    self.biases[biasIndex] = np.subtract(self.biases[biasIndex]
                                                         , biasUpdates
                                                         [biasIndex])

        print("\nCompleted training...\n")

    def backpropagate(self, actualOutput, desiredOutput, weightedInputs):
        errorValues = []
        if(self.activationFunctions[-1] == 'Softmax'):
            err = np.subtract(actualOutput, desiredOutput)
        else:
            err = np.multiply((actualOutput - desiredOutput),
                              self.activate_derivative(weightedInputs[-1],
                                                       self.activationFunctions
                                                       [-1]))
        errorValues.insert(0, err)

        for index in range(len(weightedInputs) - 2, 0, -1):
            temp = np.matmul(self.layerWeights[index],
                             np.transpose([errorValues[0]]))
            temp = np.reshape(temp, -1)
            err = np.multiply(temp,
                              self.activate_derivative(weightedInputs[index],
                                                       self.activationFunctions
                                                       [index]))
            errorValues.insert(0, err)
        return errorValues

    def activate(self, value, activationFunction):
        if isinstance(value, (list,)) or isinstance(value, (np.ndarray,)):
            newArray = []
            if(activationFunction == 'Sigmoid'):
                for elem in value:
                    newArray.append(1.0/(1.0 + np.exp(-elem)))
            if(activationFunction == 'ReLU'):
                for elem in value:
                    newArray.append(elem*(elem > 0))
            if(activationFunction == 'Tanh'):
                for elem in value:
                    newArray.append(np.tanh(elem))
            if(activationFunction == 'Softmax'):
                return self.softmax(value)
            return newArray
        else:
            if(activationFunction == 'Sigmoid'):
                return 1.0/(1.0 + np.exp(-value))
            if(activationFunction == 'ReLU'):
                return value*(value > 0)
            if(activationFunction == 'Tanh'):
                return np.tanh(value)

    def activate_derivative(self, array, activationFunction):
        newArray = []
        if(activationFunction == 'Sigmoid'):
            for elem in array:
                newArray.append(self.activate(elem, 'Sigmoid')*(1-self.activate(elem, 'Sigmoid')))
        if(activationFunction == 'ReLU'):
            for elem in array:
                if elem > 0:
                    newArray.append(1)
                else:
                    newArray.append(0)
        if(activationFunction == 'Tanh'):
            for elem in array:
                newArray.append(1 - np.square(np.tanh(elem)))
        return newArray

    def softmax(self, array):
        exp_value = np.exp(array - np.max(array))
        exp_adjusted = exp_value / np.sum(exp_value)
        return exp_adjusted
