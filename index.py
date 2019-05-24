#
# Index file
#
#

# Dependencies
from classify import Classify

# Creating object
classify = Classify()
classify.main()
# # Initializing counter for epoch
# epochCounter = 1
# # Initializing maximum number of epoch
# epochLimit = 10000
# # Initializing counter for misclassified data (Atleast one data is misclassified initially)
# misclassifiedData = 1
# # Iterating untill all data is correctly classified or maximum number of epoch is reached
# while misclassifiedData!=0 and epochCounter<epochLimit:
#     misclassifiedData = classify.main()
#     # Updating epoch counter
#     epochCounter = epochCounter+1
# # Printing no of epoch
# print("Epoch: ", epochCounter-1)
# print("Error: ", classify.calculateError())
