#
# File containing code to perform linear dicscriminant
#
#

# Dependencies
import re
import numpy
import math
import json
import matplotlib.pyplot as pyplot
import matplotlib.mlab as mlab
from helpers.readFile import ReadFile
from helpers.readCsvFile import ReadCsvFile
from helpers.getIntersection import GetIntersection
from helpers.fitCurve import FitCurve

# Initializing class
class Index:
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
        self.dataSize = config["dataSize"]
        # Setting no of features in each data
        self.noOfFeatures = config["noOfFeatures"]
        # Initializing data object (Data is created asuming equal number of training data is available for all classes)
        self.data = numpy.zeros((self.noOfClasses, self.noOfFeatures, int(self.dataSize/self.noOfClasses)))
        # Initializing counters for classes
        self.counter = numpy.zeros((self.noOfClasses))
        # Initializing discriminant vector
        self.discriminant = None
        # Initializing discriminant point
        self.discriminantPoint= None

    # Initializing function to get data
    def getData(self):
        # Getting data from file
        file = ReadCsvFile(self.fileName)
        reader = file.read()
        # Iterating over data
        for row in reader:
            # Iterating over features
            for i in range(self.noOfFeatures):
                # Storing features in data object
                self.data[int(row[3])][i][int(self.counter[int(row[3])])] = row[i+1]
            # Updating counter
            self.counter[int(row[3])]=self.counter[int(row[3])]+1

        self.calculate()

    # Initializing function to get mean
    def calculate(self):
        mean = []
        difference = []
        transpose = []
        var = []
        for i in range(self.noOfClasses):
            # Getting mean of features in i'th class
            mean.append(numpy.mean(self.data[i], axis=1))
            # Calculating Sw matrix
            difference.append(numpy.transpose(numpy.subtract(numpy.transpose(self.data[i]), numpy.transpose(mean[i]))))
            transpose.append(numpy.transpose(difference[i]))
            var.append(difference[i].dot(transpose[i]))
        # Calculating shared covariance matrix
        sw = numpy.zeros(var[0].shape)
        # Getting difference of mean
        difference = mean[0]-mean[1]
        for i in range(self.noOfClasses):
            sw = sw+var[i]
        # Calculating discriminant vector
        self.discriminant = numpy.linalg.inv(sw).dot(difference)
        # Calling function to reduce dimensions
        self.reduceDimension()

    # Initializing function to reduce data to single dimension
    def reduceDimension(self):
        # Converting data into single dimension
        class1 = self.discriminant.dot(self.data[0])
        class2 = self.discriminant.dot(self.data[1])
        # Calling plotPoints function
        self.plotPoints(class1, class2)

    # Initializing function to plot data
    def plotPoints(self, class1, class2):
        # Creating dummy array for y-axis
        y = numpy.zeros(int(self.dataSize/self.noOfClasses))
        # Getting parameters to fit data into normal distribution
        class1Parameters = FitCurve(class1).getParameters()
        class2Parameters = FitCurve(class2).getParameters()
        # Plotting class 1 points
        pyplot.plot(class1, y, "ro", markerfacecolor='none')
        # Plotting class 1 points
        pyplot.plot(class2, y, "go", markerfacecolor='none')
        # Plotting gaussian curve for class 1
        x = numpy.linspace(class1Parameters["mean"] - 3*math.sqrt(class1Parameters["var"]), class1Parameters["mean"] + 3*math.sqrt(class1Parameters["var"]), int(self.dataSize/self.noOfClasses))
        pyplot.plot(sorted(class1), mlab.normpdf(x, class1Parameters["mean"], math.sqrt(class1Parameters["var"])))
        # Plotting gaussian curve for class 2
        x = numpy.linspace(class2Parameters["mean"] - 3*math.sqrt(class2Parameters["var"]), class2Parameters["mean"] + 3*math.sqrt(class2Parameters["var"]), int(self.dataSize/self.noOfClasses))
        pyplot.plot(sorted(class2), mlab.normpdf(x, class2Parameters["mean"], math.sqrt(class2Parameters["var"])))
        # Getting point of intersection of curves
        self.discriminantPoint = GetIntersection(class1Parameters, class2Parameters).getResult()
        # Displaying plot
        pyplot.plot(self.discriminantPoint, 0, "bo")
        pyplot.show()

# Creating object
index  = Index()
index.getData()
