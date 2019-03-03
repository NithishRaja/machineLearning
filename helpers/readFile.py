#
# File containing code to read file
#
#

# Initializing class
class ReadFile:
    # Initializing constructor
    def __init__(self, fileName):
        # Setting fileName
        self.fileName = fileName

    def read(self):
        file = None
        try:
            # Reading from file
            file = open(self.fileName, "r")
        except FileNotFoundError:
            print("file not found at "+self.fileName)
            # Returning false to indicate error
            return False
        else:
            # Reading data from file
            return file.read()
