#
# File containing code to implement perceptron
#
#

# Dependencies
import json
import os
import re
import operator
from helpers.readFile import ReadFile
from parse import Parse

# Initializing class
class Classify:
    # Initializing constructor
    def __init__(self):
        # Reading config file
        file = ReadFile("config.json")
        config = json.loads(file.read())
        # Setting data size
        self.dataSize = config["dataSize"]
        # Setting data file name
        self.dataFileName = config["dataFileName"]
        # Setting frequency file name
        self.frequencyFileName = config["frequencyFileName"]
        # Setting training set split
        self.trainingSetSize = int(self.dataSize*config["trainingSetPercentage"]/100)
        # Setting validation set split
        self.validationSetSize = self.dataSize - self.trainingSetSize
        # Initializing array to hold all class names
        self.classes = config["class"]
        # Checking if frequency data is already available
        if not os.path.exists(self.frequencyFileName):
            # Getting frequency data
            parse = Parse(self.dataFileName, self.frequencyFileName, self.trainingSetSize, self.classes)
            parse.main()
        # Setting number of data available
        self.dataSize = config["dataSize"]
        # Reading frequency data file
        file = ReadFile(self.frequencyFileName)
        self.frequencyData = json.loads(file.read())
        # Initializing object to hold probability for positive and negative statements
        self.probability = {}
        # Setting probability of each class
        for className in self.classes:
            self.probability[className] = self.frequencyData[className+"Frequency"]/self.trainingSetSize
        # Initializing object to hold values for confusion matrix
        self.confusionMatrix = {}
        for className in self.classes:
            names = {}
            for name in self.classes:
                names[name] = 0
            self.confusionMatrix[className] = names

    # Function to classify data
    def main(self):
        # Initializing variable to hold fileDescriptor
        file = None
        try:
            # Opening file
            file = open(self.dataFileName, "r", encoding="utf8")
        except FileNotFoundError:
            # Printing error
            print("file not found at "+self.dataFileName)
        else:
            # Initializing counter for getting number of data instances
            counter = 0
            # iterating over each line in data
            for line in file:
                # Updating counter
                counter = counter + 1
                # Run code only for validation set i.e., last 20 % of data
                if counter>self.trainingSetSize:
                    # Splitting line into separate words
                    words = re.split(" ", line)
                    # Iterating over each word in current line
                    for i in range(3, len(words)):
                        # Calling function to get conditional probability of current word
                        self.getConditionalProbability(words[i])
                    # Updating confusion matrix
                    actualClass = words[1]
                    predictedClass = max(self.probability.items(), key=operator.itemgetter(1))[0]
                    self.confusionMatrix[actualClass][predictedClass] = self.confusionMatrix[actualClass][predictedClass]+1
            # Closing file
            file.close()
            # Returning confusion matrix
            return self.confusionMatrix

    # Function to get conditional frequency of given word
    def getConditionalProbability(self, word):
        # Iterating over classes
        for className in self.classes:
            # Checking if word occured in training set, if word has not occcured ignore it
            if word in self.frequencyData[className]:
                # Word exists and probability is calculated
                self.probability[className] = self.probability[className]*self.frequencyData[className][word]/self.frequencyData[className+"Frequency"]
