#
# File containing code to perform linear dicscriminant
#
#

# Dependencies
import numpy
import json
import matplotlib.pyplot as pyplot
import matplotlib.mlab as mlab
from helpers.readFile import ReadFile
from helpers.readCsvFile import ReadCsvFile
from helpers.getIntersection import GetIntersection
from helpers.fitCurve import FitCurve

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
        # Initializing discriminant vector
        self.discriminant = None
        # Initializing discriminant point
        self.discriminantPoint= None
        # Calling function to get input data
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

    # Initializing function to get mean
    def main(self):
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
        classfication = []
        # Converting data into single dimension
        for i in range(self.noOfClasses):
            classfication.append(self.discriminant.dot(self.data[i]))
        # Calling plotPoints function
        self.plotPoints(classfication)

    # Initializing function to plot data
    def plotPoints(self, classfication):
        classParams = []
        # Creating dummy array for y-axis
        y = numpy.zeros(int(self.dataSize/self.noOfClasses))
        # Getting parameters to fit data into normal distribution
        for i in range(self.noOfClasses):
            classParams.append(FitCurve(classfication[i]).getParameters())
        # Plotting data points
        marker = ["ro", "go"]
        for i in range(self.noOfClasses):
            pyplot.plot(classfication[i], y, marker[i], markerfacecolor='none')
        # Plotting gaussian curve for each class
        for i in range(self.noOfClasses):
            x = numpy.linspace(classParams[i]["mean"] - 3*classParams[i]["sd"], classParams[i]["mean"] + 3*classParams[i]["sd"], int(self.dataSize/self.noOfClasses))
            pyplot.plot(x, mlab.normpdf(x, classParams[i]["mean"], classParams[i]["sd"]))
        # Getting point of intersection of curves
        self.discriminantPoint = GetIntersection(classParams[0], classParams[1]).getResult()
        # Plotting discriminating point
        pyplot.plot(self.discriminantPoint, 0, "bo")
        # Saving plot as an image
        pyplot.savefig("plot/"+self.fileName+"_plot.png")


    # Initializing function to calculate error
    def calculateError(self):
        # Initializing counter for misclassified points
        misclassifiedPoints = 0
        # Array to hold reduced values of each data instance
        classfication = []
        # Converting data into single dimension
        for i in range(self.noOfClasses):
            classfication.append(self.discriminant.dot(self.data[i]))
            # Iterating over all values
            for j in classfication[i]:
                # Checking if point has been misclassified
                if (j>self.discriminantPoint and i==1) or (j<self.discriminantPoint and i==0):
                    misclassifiedPoints = misclassifiedPoints+1
        # Printing number of misclassified points and error %
        print("No of misclassified points: ", misclassifiedPoints)
        print("Error%: ", (misclassifiedPoints/self.dataSize)*100)
