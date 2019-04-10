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
        self.trainingSetPercentage = config["trainingSetPercentage"]
        # Setting validation set split
        self.validationSetPercentage = 100 - self.trainingSetPercentage
        # Checking if frequency data is already available
        if !os.path.exists(self.frequencyFileName):
            # Getting frequency data
            parse = Parse(self.dataFileName, self.frequencyFileName, self.dataSize, self.trainingSetPercentage)
            parse.main()
        # Setting number of data available
        self.dataSize = config["dataSize"]

    # Function to classify data
    def main(self):
