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

    # Function to classify data
    # def main(self):
