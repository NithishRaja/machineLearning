#
# Index file
#
#

# Dependencies
from classify import Classify

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
