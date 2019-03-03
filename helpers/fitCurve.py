#
# File containing code to fit data into a gaussian distribution
#
#

# Dependencies
import numpy
import math

# Initializing class
class FitCurve:
    # Initializing constructor
    def __init__(self, data):
        # Storing data
        self.data = data

    def getParameters(self):
        # Initializing variables
        mean = 0
        var = 0
        sd = 0
        # Setting mode of data as mean of distribution
        mean = numpy.mean(self.data)
        # Calculating variance
        # Iterating over data
        for i in self.data:
            var = var+(math.pow((i-mean), 2))/len(self.data)
        # Calculating standard deviation
        sd = math.sqrt(var)
        # Returning data
        return {"mean":mean, "var":var, "sd":sd}
