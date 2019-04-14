#
# File containing code to calculate sigmoid function
#
#

# Dependencies
import math

# Initializing sigmoid function
def sigmoid(value):
    # Calculating function result
    result = 1 / (1 + math.exp(-value))
    # Returning result
    return result
