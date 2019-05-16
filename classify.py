#
# File containing code to implement perceptron
#
#

# Dependencies
import numpy
import json
import csv
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
        self.layerTwoWeights = numpy.random.rand(self.noOfNodes)
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
    # def main(self):


    # Initializing function to calculate error %
    # def calculateError(self):
