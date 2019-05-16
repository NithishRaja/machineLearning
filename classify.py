#
# File containing code to implement perceptron
#
#

# Dependencies
import numpy
import json
import csv
import math
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
        # Initializing mean value for each class with a random value
        self.mean = numpy.random.rand(self.noOfClasses, self.noOfFeatures)
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
            self.data.append(numpy.zeros((int(self.counter[i]), self.noOfFeatures), dtype=float))
        # Resetting file to top
        file.seek(0)
        # Resetting counter
        self.counter = numpy.zeros((self.noOfClasses), dtype=int)
        # Storing data instances in self.data
        for row in reader:
            # Iterating over features
            for i in range(self.noOfFeatures):
                # Storing features in data object
                self.data[int(row[-1])][int(self.counter[int(row[-1])])][i] = row[i]
            # Updating counter
            self.counter[int(row[-1])]=self.counter[int(row[-1])]+1

    # Function to classify data
    def main(self):
        # Initializing array to hold cluster size
        self.clusterSize = []
        # Initializing cluster for each class
        self.cluster = []
        # Setting each cluster to an empty array
        for i in range(self.noOfClasses):
            self.cluster.append([])
            self.clusterSize.append(0)
        # Iterating over each class
        for i in range(self.noOfClasses):
            # Iterating over data in each class
            for j in range(self.counter[i]):
                # Calling function to calculate distance
                result = self.calculateDistance(self.data[i][j])
                # Setting data in cluster corresponding to result
                self.cluster[result].append(self.data[i][j])
                # Updating cluster size
                self.clusterSize[result] = self.clusterSize[result] + 1
        # Calling function to calculate new mean
        distance = self.calculateNewMean()
        # Returning maximum distance
        return max(distance)

    # Function to calculate distance of data from each mean and return minimum
    def calculateDistance(self, data):
        # Initializing variable to hold minimum distance
        min = None
        minClass = None
        # Iterating over each class
        for i in range(self.noOfClasses):
            # Calculating distance
            dist = math.sqrt(sum(numpy.square(self.mean[i]-data)))
            # Checking if minimum distance has a value
            if min == None:
                min = dist
                minClass = i
            elif dist < min:
                min = dist
                minClass = i
        # Returning class label with minimum distance
        return minClass

    # Function to calculate new mean
    def calculateNewMean(self):
        # Initializing variable to hold distance between new mean and current mean
        distance = numpy.zeros((self.noOfClasses))
        # Initializing variable to hold values for new mean
        newMean = numpy.random.rand(self.noOfClasses, self.noOfFeatures)
        # Iterating over each class
        for i in range(self.noOfClasses):
            newMean[i] = numpy.divide(sum(self.cluster[i]), self.clusterSize[i])
            # Clculating distance between new mean and previous mean
            distance[i] = math.sqrt(sum(numpy.square(self.mean[i]-newMean[i])))
        # Updating mean
        self.mean = newMean
        # Returning distance
        return distance

    # Initializing function to calculate error %
    def calculateError(self):
        error = 0
        # Iterating over each class
        for i in range(self.noOfClasses):
            # Iterating over each data in current class
            for j in range(self.counter[i]):
                # Calling function to calculate distance
                result = self.calculateDistance(self.data[i][j])
                # Checking if data is misclassified
                if result != i:
                    # Updating error
                    error = error + 1
        return error
