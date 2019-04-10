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
        # Setting file name
        self.dataFileName = config["dataFileName"]
        self.frequencyFileName = config["frequencyFileName"]
        # Checking if frequency data is already available
        if !os.path.exists(self.frequencyFileName):
            # Getting frequency data
            parse = Parse(self.dataFileName, self.frequencyFileName)
            parse.main()
        # Setting number of data available
        self.dataSize = config["dataSize"]

    # Function to classify data
    # def main(self):
