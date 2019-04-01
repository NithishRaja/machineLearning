#
# File containing code to implement perceptron
#
#

# Dependencies
import numpy
import json
import matplotlib.pyplot as pyplot
from helpers.readFile import ReadFile
from helpers.readCsvFile import ReadCsvFile

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
        self.dataSize = config["dataSize"]
        # Setting no of features in each data
        self.noOfFeatures = config["noOfFeatures"]
        # Initializing data object (Data is created asuming equal number of training data is available for all classes)
        self.data = numpy.zeros((self.noOfClasses, self.noOfFeatures, int(self.dataSize/self.noOfClasses)))
        # Initializing counters for classes
        self.counter = numpy.zeros((self.noOfClasses))
        # Setting learning rate
        self.learningRate = config["learningRate"]
        # Initializing weight vector
        self.weightVector = numpy.arange(self.noOfFeatures+1, dtype=float)
        # Calling function to get data
        self.getData()

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

    # Function to classify data
    def main(self):
        # Counter for misclassified data
        misclassifiedData = 0
        # Iterating over each class
        for i in range(self.noOfClasses):
            # Iterating over data in each class
            for j in range(int(self.dataSize/self.noOfClasses)):
                value = self.weightVector[self.noOfFeatures]
                # Iterating over each feature
                for k in range(self.noOfFeatures):
                    value = value + self.data[i][k][j]*self.weightVector[k]
                # Checking if data is misclassified
                if value*(i-0.5)<0:
                    # Update counter for misclassified data
                    misclassifiedData = misclassifiedData+1
                    # Updating weight vector
                    self.weightVector[self.noOfFeatures] = self.weightVector[self.noOfFeatures]+self.learningRate*2*(i-0.5)
                    for k in range(self.noOfFeatures):
                        self.weightVector[k] = self.weightVector[k]+self.learningRate*2*(i-0.5)*self.data[i][k][j]
        # Returning no of misclassified data
        return misclassifiedData

    # Initializing function to plot data
    def plotPoints(self, plotName, title):
        # Plotting data points
        # Setting class properties
        marker = ["ro", "go"]
        # Setting title
        pyplot.title(title)
        # Iterating over each class
        for i in range(self.noOfClasses):
            # Iterating over data in each class
            for j in range(int(self.dataSize/self.noOfClasses)):
                # If first data in this class, add label
                if j==0:
                    pyplot.plot(self.data[i][0][j], self.data[i][1][j], marker[i], markerfacecolor='none', label="class "+str(i))
                else:
                    pyplot.plot(self.data[i][0][j], self.data[i][1][j], marker[i], markerfacecolor='none')
        # Plotting hyperplane to separate classes
        x = numpy.linspace(-3, 3, 3)
        y = -(self.weightVector[0]*x+self.weightVector[2])/self.weightVector[1]
        pyplot.plot(x, y)
        # Setting label
        pyplot.legend()
        # Saving plot as an image file
        pyplot.savefig(plotName)
        # Closing plot
        pyplot.close()

    # Initializing function to calculate error %
    def calculateError(self):
        # Initializing counter for misclassified data
        misclassifiedData = 0
        # Iterating over each class
        for i in range(self.noOfClasses):
            # Iterating over data in each class
            for j in range(int(self.dataSize/self.noOfClasses)):
                value = self.weightVector[self.noOfFeatures]
                # Iterating over each feature
                for k in range(self.noOfFeatures):
                    value = value + self.data[i][k][j]*self.weightVector[k]
                # Checking if data is misclassified
                if (value<0 and i==1) or (value>0 and i==0):
                    misclassifiedData = misclassifiedData+1
        # Printing number of misclassified data and error %
        print("No of misclassified data: ", misclassifiedData)
        print("Error %: ", (misclassifiedData/self.dataSize)*100)

    # Function to classify new point
    def classifyPoint(self, point):
        distance = 0.0
        # Calculating product between w vector and point
        for i in range(self.noOfFeatures):
            distance = distance + point[i]*self.weightVector[i]
        # Adding bias
        distance = distance + self.weightVector[self.noOfFeatures]
        # Checking if data belongs to class 0 or class 1
        if distance<0:
            print("Class 0")
        else:
            print("Class 1")
