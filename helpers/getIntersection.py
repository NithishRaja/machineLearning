#
# File containing code to get roots of quadratic equation
#
#

# Dependencies
import numpy

# Initializing class
class GetIntersection:
    def __init__(self, class1Parameters, class2Parameters):
        # Storing mean values
        self.mean1 = class1Parameters["mean"]
        self.mean2 = class2Parameters["mean"]
        # Calculating coefficients
        self.a = 1/(2*class1Parameters["sd"]**2) - 1/(2*class2Parameters["sd"]**2)
        self.b = class2Parameters["mean"]/(class2Parameters["sd"]**2) - class1Parameters["mean"]/(class1Parameters["sd"]**2)
        self.c = class1Parameters["mean"]**2 /(2*class1Parameters["sd"]**2) - class2Parameters["mean"]**2 / (2*class2Parameters["sd"]**2) - numpy.log(class2Parameters["sd"]/class1Parameters["sd"])
        # Initializing result to None
        self.result = None

    def getResult(self):
        # Getting roots of quadratic equation
        roots = numpy.roots([self.a, self.b, self.c])
        # Getting point lying between both means
        for i in roots:
            if (i-self.mean1)*(i-self.mean2)<0:
                self.result = i
        return self.result
