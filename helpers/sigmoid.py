#
# File containing code to calculate sigmoid function
#
#

# Dependencies
import math

# Initializing sigmoid function
def sigmoid(value):
    ans = 0
    try:
        ans = math.exp(-value)
    except OverflowError:
        ans = float('inf')
    # Calculating function result
    result = 1 / (1 + ans)
    # Returning result
    return result
