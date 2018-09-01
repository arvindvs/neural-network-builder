import numpy as np


class NeuralNetwork:
    """
    Neural network implementation

    Functions:
        _init_
        feed_forward
        train
        backpropagate
        activate
        activate_derivative
        softmax

    Attributes:
        inputSize: number of inputs expected
        outputSize: number of possible classifications
        layerWeights: list of matrices representing weights between layers
        biases: list of vectors representing bias before activation
        activationFunctions: list of activation functions at each layer
        trainingMethod: string containing gradient descent approach
        learningRate: float representing learning rate in training
    """

    def __init__(self, layerSizes, learningRate, activationFunction='ReLU',
                 trainingMethod='batchGradientDescent'):
        """Initializes NeuralNetwork object.

        Defines all the pertinent instance variables required to construct and
        manipulate a NeuralNetwork object.

        Args:
            layerSizes: A list of ints representing the size of each layer.
            learningRate: A float that defines the rate of gradient descent.
            activationFunction: An optional string that defines the activation
                function used in each hidden layer. Defaults to 'ReLU'.
            trainingMethod: An optional string defining approach to gradient
                descent. Defaults to 'batchGradientDescent'.

        Returns:
            None
        """
        self.inputSize = layerSizes[0]
        self.outputSize = layerSizes[-1]

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
        """Sends input through each layer of the network.

        Yields the set of outputs computed by a forward-pass through the neural
        network. Defines weighted inputs to be used in training if TrainFlag is
        turned on.

        Args:
            input: A list of ints representing the size of each layer.
            trainFlag: An optional boolean that determines if forward-pass is
                being used for training purposes.

        Returns:
            A list containing the output values.
        """
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
        """Trains network using supervised learning method approach.

        Yields a set of updated weights and biases that have improved accuracy
        in feed-forward inference.

        Args:
            inputs: A list of arrays representing each input data point. Input
                must be the same size as the first layer.
            outputs: A list of arrays representing each output data point.
                Each output array must have the same size as the number of
                classes.
            epochs: Optional integer that determines number of gradient descent
                iterations.

        Returns:
            None
        """
        print("Starting training...\n\n")
        assert len(inputs) == len(outputs)

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
        """Performs a single iteration of backpropagation by calculating the
        error in each layer.

        Yields the set of errors in each layer based on an experimental output
        and desired output. These errors are then used to modify the weights
        and biases of the network according to the desired gradient descent
        method.

        Args:
            actualOutput: An array containing the values from a forward pass
            desiredOutput: An array containing the expected/desired values
                based on the corresponding input values
            weightedInputs: A list of arrays representing the value at each
                layer calculated during forward propagation before activation.
                Passed as a parameter for the sake of convenience so the values
                don't have to be calculated multiple times.

        Returns:
            A list of arrays representing the error in each layer
        """
        errorValues = []

        # Softmax output uses Cross-Entropy Loss while others use Square Error
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
        """Modifies a value using the given activation function

        Activates the value, provided as either an array or single input, using
        the user-inputted activation function.
        This activate function is called in feed_forward and backpropagate.

        Args:
            value: Either an array or single float value representing value
                to be activated.
            activationFunction: A string representing the user-inputted
                function to be used on the given value.

        Returns:
            Activated value
        """
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
        """Derivative of the activation function

        Used in the backpropagate function to calculate error at each layer.

        Args:
            value: An array representing the values to be activated.
            activationFunction: A string representing the user-inputted
                function whose derivative is to be used on the given value.
        Returns:
            Derivative of activated value
        """
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
        """Performs softmax on each value of a given array.

        A normalizing function that makes each value in array add to 1. Useful
        in classification for determining probability of each class. Used in
        the activate function when called on an array. Normally called on the
        last layer of the network.

        Args:
            array: An array to be used in the softmax function. Normally
                represents the output layer of a network.

        Returns:
            Softmax of array
        """
        exp_value = np.exp(array - np.max(array))
        exp_adjusted = exp_value / np.sum(exp_value)
        return exp_adjusted
