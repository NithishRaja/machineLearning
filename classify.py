#
# File containing code to implement perceptron
#
#

# Dependencies
import numpy
import json
import csv
import math
from helpers.readFile import ReadFile
from helpers.sigmoid import sigmoid

# Initializing class
class Classify:
    # Initializing constructor
    def __init__(self):
        # Reading config file
        file = ReadFile("config.json")
        config = json.loads(file.read())
        # Setting file name
        self.fileName = config["fileName"]
        # Setting data size
        self.dataSize = config["dataSize"]
        # Setting no of classes
        self.noOfClasses = config["noOfClasses"]
        # Setting no of features in each data
        self.noOfFeatures = config["noOfFeatures"]
        # Setting number of nodes
        self.noOfNodes = config["noOfNodes"]
        # Initializing data object
        self.data = numpy.zeros((self.dataSize, self.noOfFeatures))
        # Initializing array to store expected output
        self.expectedOutput = numpy.zeros(self.dataSize)
        # Initializing counters for classes
        self.counter = 0
        # Setting learning rate
        self.learningRate = config["learningRate"]
        # Initializing first layer weights
        self.layerOneWeight = numpy.random.rand(self.noOfFeatures, self.noOfNodes)
        # Initializing second layer weights
        self.layerTwoWeight = numpy.random.rand(self.noOfNodes)
        # Calling function to get data
        self.getData()

    # Initializing function to get data
    def getData(self):
        # Getting data from file
        file = open(self.fileName, "r")
        reader = csv.reader(file)
        # Iterating over data to get no of instances for each class
        for row in reader:
            # Iterating over each feature
            for i in range(self.noOfFeatures):
                # Storing data
                self.data[self.counter][i] = row[i]
            # Storing expected output
            self.expectedOutput[self.counter] = row[-1]
            # Updating counter
            self.counter = self.counter + 1

    # Function to classify data
    def main(self):
        # Calculating layer one input
        self.layerOneInput = self.data@self.layerOneWeight
        # Calculating layer one output
        self.layerOneOutput = numpy.vectorize(self.sigmoid)(self.layerOneInput)
        # Calculating layer two input
        self.layerTwoInput = self.layerOneOutput@self.layerTwoWeight
        # Calculating layer two output
        self.layerTwoOutput = numpy.vectorize(self.sigmoid)(self.layerTwoInput)
        # Getting class labels from probabilities
        self.result = numpy.vectorize(self.round)(self.layerTwoOutput)

    # Initializing function to calculate error %
    def calculateError(self):
        misclassifiedData = 0
        # Calculating error
        self.error = self.result - self.expectedOutput
        # Iterating over each data
        for i in range(self.dataSize):
            # Checking if calculated class and actual class match
            if not self.error[i] == 0:
                misclassifiedData = misclassifiedData + 1
        # Return no of misclassified data
        return misclassifiedData

    # Function to update weight vector
    def updateWeight(self):
        # Calculating weight update for layer two
        self.layerTwoUpdate = self.error@self.layerOneOutput
        # Calculating backpropagated error
        backpropagatedError = sum(self.layerTwoWeight.reshape(self.layerTwoWeight.shape[0], 1)@self.error.reshape(1, self.error.shape[0]))
        # Calculating weight update for layer one
        self.layerOneUpdate = numpy.transpose(self.data)@numpy.multiply(numpy.vectorize(self.sigmoidDerivative)(self.layerOneInput), backpropagatedError.reshape(backpropagatedError.shape[0], 1))
        # Updating layer two weights
        self.layerTwoWeight = self.layerTwoWeight - self.learningRate*self.layerTwoUpdate
        # Updating layer one weights
        self.layerOneWeight = self.layerOneWeight - self.learningRate*self.layerOneUpdate

    # Function to calculate sigmoid value
    def sigmoid(self, value):
        # Returning sigmoid value
        return 1/(1+math.exp(-value))

    # Function to calculate derivative of sigmoid function
    def sigmoidDerivative(self, value):
        # Getting sigmoid value
        result = self.sigmoid(value)
        # Returning derivative value
        return result*(1-result)

    # Function to convert probability into class labels
    def round(self, value):
        # Return class label depending upon probability
        if value > 0.5:
            return 1
        else:
            return 0
