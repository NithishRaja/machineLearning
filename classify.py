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
        # Setting no of classes
        self.noOfClasses = config["noOfClasses"]
        # Setting number of data available
        self.dataSize = 0
        # Setting no of features in each data
        self.noOfFeatures = config["noOfFeatures"]
        # Initializing data object
        self.data = []
        # self.data = numpy.zeros((self.noOfClasses, self.noOfFeatures, int(self.dataSize/self.noOfClasses)))
        # Initializing counters for classes
        self.counter = numpy.zeros((self.noOfClasses))
        # Setting learning rate
        self.learningRate = config["learningRate"]
        # Initializing weight vector
        self.weightVector = numpy.arange(self.noOfFeatures, dtype=float)
        # Calling function to get data
        self.getData()

    # Initializing function to get data
    def getData(self):
        # Getting data from file
        file = open(self.fileName, "r")
        reader = csv.reader(file)
        # Iterating over data to get no of instances for each class
        for row in reader:
            # Updating counter
            self.counter[int(row[-1])]=self.counter[int(row[-1])]+1
        # Setting Array size for data
        for i in range(self.noOfClasses):
            self.data.append(numpy.zeros((self.noOfFeatures, int(self.counter[i])), dtype=float))
        # Resetting file to top
        file.seek(0)
        # Setting data size
        self.dataSize = sum(self.counter)
        # Resetting counter
        self.counter = numpy.zeros((self.noOfClasses))
        # Storing data instances in self.data
        for row in reader:
            # Iterating over features
            for i in range(self.noOfFeatures):
                # Storing features in data object
                self.data[int(row[-1])][i][int(self.counter[int(row[-1])])] = row[i]
            # Updating counter
            self.counter[int(row[-1])]=self.counter[int(row[-1])]+1

    # Function to classify data
    # def main(self):

    # Initializing function to calculate error %
    # def calculateError(self):
