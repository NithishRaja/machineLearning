#
# File containing code to get frequency of each word
#
#

# Dependencies
import re
import json

# Initializing class
class Parse:
    # Initializing constructor
    def __init__(self, dataFileName, frequencyFileName):
        # Setting data file path
        self.dataFileName = dataFileName
        self.frequencyFileName = frequencyFileName
        # Initializing object to hold word frequency
        self.frequency = {
            "pos": {},
            "neg": {}
        }

    # Function to read file data
    def getData(self):
        # Initializing variable to hold fileDescriptor
        file = None
        try:
            # Opening file
            file = open(self.dataFileName, "r", encoding="utf8")
        except FileNotFoundError:
            # Printing error
            print("file not found at "+self.dataFileName)
        else:
            # iterating over each line in data
            for line in file:
                words = re.split(" ", line)
                # Iterating over each word in current line
                for i in range(3, len(words)):
                    # Update frequency of word
                    if words[i] in self.frequency[words[1]]:
                        self.frequency[words[1]][words[i]] = self.frequency[words[1]][words[i]]+1
                    else:
                        self.frequency[words[1]][words[i]] = 1
            # Closing file
            file.close()

    # Persist frequency data to file
    def writeFrequencyData(self):
        # Initializing variable to hold fileDescriptor
        file = None
        try:
            # Opening file
            file = open(self.frequencyFileName, "w")
        except FileNotFoundError:
            # Printing error
            print("file not found at "+self.frequencyFileName)
        else:
            # Writing data to file
            json.dump(self.frequency, file)
            # Closing file
            file.close()
