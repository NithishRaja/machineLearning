#
# File containing code to perform linear dicscriminant
#
#

# Dependencies
import re
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.mlab as mlab
import json
from helpers.readFile import ReadFile
from helpers.getIntersection import GetIntersection
from helpers.fitCurve import FitCurve

# Initializing class
class Index:
    # Initializing constructor
    def __init__(self):
        # Reading config file
        file = ReadFile("config.json")
        config = json.loads(file.read())
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
        file = ReadFile("data.txt")
        dataString = file.read()
        # Splitting data
        dataList = re.split("\n", dataString)
        # Removing final empty entry from dataList
        dataList.pop()
        # Iterating through data list
        for data in dataList:
            # Getting features from data
            features = re.split(",", data)
            # Checking if features belong to class 1
            if features[4]=="Iris-setosa":
                # Features belong to class 1, update class1 object with features
                # Iterating through features
                for i in range(self.noOfFeatures):
                    # Setting value in data
                    self.data[0][i][int(self.counter[0])] = features[i]
                # Updating counter
                self.counter[0]=self.counter[0]+1
            else:
                # Features belong to class 2, update class1 object with features
                # Iterating through features
                for i in range(self.noOfFeatures):
                    # Setting value in data
                    self.data[1][i][int(self.counter[1])] = features[i]
                # Updating counter
                self.counter[1]=self.counter[1]+1
        self.calculate()

    # Initializing function to get mean
    def calculate(self):
        # Getting mean of features in class 1
        self.mean1 = numpy.mean(self.data[0], axis=1)
        # Getting mean of features in class 2
        self.mean2 = numpy.mean(self.data[1], axis=1)
        # Calculating Sw matrix
        difference1 = numpy.transpose(numpy.subtract(numpy.transpose(self.data[0]), numpy.transpose(self.mean1)))
        difference2 = numpy.transpose(numpy.subtract(numpy.transpose(self.data[1]), numpy.transpose(self.mean2)))
        transpose1 = numpy.transpose(difference1)
        transpose2 = numpy.transpose(difference2)
        var1 = difference1.dot(transpose1)
        var2 = difference2.dot(transpose2)
        self.sw = var1+var2
        # Getting difference of mean
        self.difference = self.mean2-self.mean1
        # Calculating discriminant vector
        self.discriminant = numpy.linalg.inv(self.sw).dot(self.difference)
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
        x = numpy.linspace(class1Parameters["mean"] - 3*class1Parameters["var"], class1Parameters["mean"] + 3*class1Parameters["var"], 100)
        pyplot.plot(x, mlab.normpdf(x, class1Parameters["mean"], class1Parameters["var"]))
        # Plotting gaussian curve for class 2
        x = numpy.linspace(class2Parameters["mean"] - 3*class2Parameters["var"], class2Parameters["mean"] + 3*class2Parameters["var"], 100)
        pyplot.plot(x, mlab.normpdf(x, class2Parameters["mean"], class2Parameters["var"]))
        # Getting point of intersection of curves
        self.discriminantPoint = GetIntersection(class1Parameters, class2Parameters).getResult()
        # Displaying plot
        pyplot.plot(self.discriminantPoint, 0, "bo")
        pyplot.show()

# Creating object
index  = Index()
index.getData()
