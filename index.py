#
# Index file
#
#

# Dependencies
from classify import Classify

# Creating object
classify = Classify()
classify.main()
# Initializing counter for epoch
epochCounter = 1
# Initializing maximum number of epoch
epochLimit = 1000
# Initializing counter for misclassified data (Atleast one data is misclassified initially)
misclassifiedData = classify.calculateError()
# Iterating untill all data is correctly classified or maximum number of epoch is reached
while misclassifiedData!=0 and epochCounter<epochLimit:
    # Calling function to update weights
    classify.updateWeight()
    # Calling function to classify data
    classify.main()
    # Calling functon to calculate no of misclassified points
    misclassifiedData = classify.calculateError()
    # Updating epoch counter
    epochCounter = epochCounter+1
# Printing no of epoch
print("Epoch: ", epochCounter-1)
print("Error: ", classify.calculateError())
