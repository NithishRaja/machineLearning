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
    def __init__(self, dataFileName, frequencyFileName, trainingSetSize):
        # Setting data file path
        self.dataFileName = dataFileName
        # Setting frequency dile path
        self.frequencyFileName = frequencyFileName
        # Setting size of data to use for training data
        self.trainingSetSize = trainingSetSize
        # Initializing object to hold word frequency
        self.frequency = {
            "pos": {},
            "neg": {}
        }

    # Function to read file data
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
                # Splitting line into separate words
                words = re.split(" ", line)
                # Iterating over each word in current line
                for i in range(3, len(words)):
                    # Update frequency of word
                    if words[i] in self.frequency[words[1]]:
                        self.frequency[words[1]][words[i]] = self.frequency[words[1]][words[i]]+1
                    else:
                        self.frequency[words[1]][words[i]] = 1
                # Exit loop if trainingSetSize is reached
                if counter>=self.trainingSetSize:
                    break
            # Closing file
            file.close()
        # Calling function to persist frequency data
        self.writeFrequencyData()

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
