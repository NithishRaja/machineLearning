#
# Index file
#
#

# Dependencies
import os
from classify import Classify

# Checking if plot directory exists,
# If it doesn't exist create it
if not os.path.exists("plot/"):
    os.mkdir("plot/")

# Creating object
classify = Classify()
# Plotting initial hyperplane
classify.plotPoints("plot/initial_plot.png")
# Initializing counter for epoch
epochCounter = 1
# Initializing maximum number of epoch
epochLimit = 500
# Initializing counter for misclassified data (Atleast one data is misclassified initially)
misclassifiedData = 1
# Iterating untill all data is correctly classified or maximum number of epoch is reached
while misclassifiedData!=0 and epochCounter<epochLimit:
    misclassifiedData = classify.main()
    # classify.plotPoints("plot/epoch_"+str(epochCounter)+".png")
    epochCounter = epochCounter+1
# Printing error %
classify.calculateError()
