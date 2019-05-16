#
# File containing code to implement perceptron
#
#

# Dependencies
import numpy
import json
import csv
import matplotlib.pyplot as pyplot
from helpers.readFile import ReadFile

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
        # Setting no of features in each data
        self.noOfFeatures = config["noOfFeatures"]
        # Initializing data object (Data is created asuming equal number of training data is available for all classes)
        self.data = []
        # Initializing counters for classes
        self.counter = numpy.zeros((self.noOfClasses))
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
            self.counter[int(row[0])]=self.counter[int(row[0])]+1
        # Setting Array size for data
        for i in range(self.noOfClasses):
            self.data.append(numpy.zeros((int(self.counter[i]), self.noOfFeatures), dtype=float))
        # Resetting file to top
        file.seek(0)
        # Resetting counter
        self.counter = numpy.zeros((self.noOfClasses))
        # Storing data instances in self.data
        for row in reader:
            # Iterating over features
            for i in range(self.noOfFeatures):
                # Storing features in data object
                self.data[int(row[0])][int(self.counter[int(row[0])])][i] = row[i+1]
            # Updating counter
            self.counter[int(row[0])]=self.counter[int(row[0])]+1

    # Function to classify data
    # def main(self):

    # Initializing function to calculate error %
    # def calculateError(self):

    # Function to classify new point
    # def classifyPoint(self, point):
