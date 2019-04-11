#
# File containing code to implement perceptron
#
#

# Dependencies
import json
import os
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
        # Checking if frequency data is already available
        if not os.path.exists(self.frequencyFileName):
            # Getting frequency data
            parse = Parse(self.dataFileName, self.frequencyFileName, self.trainingSetSize)
            parse.main()
        # Setting number of data available
        self.dataSize = config["dataSize"]
        # Reading frequency data file
        file = ReadFile(self.frequencyFileName)
        self.frequencyData = json.loads(file.read())
        # Initializing object to hold probability for positive and negative statements
        self.prob = {
            "pos": self.frequencyData["posFrequency"]/self.trainingSetSize,
            "neg": self.frequencyData["negFrequency"]/self.trainingSetSize
        }

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
                        self.getConditionalFreqency(word[i])
            # Closing file
            file.close()
            # Calling function to persist frequency data

    # Function to get conditional frequency of given word
    # def getConditionalFreqency(self, word):
