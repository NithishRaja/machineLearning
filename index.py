#
# Index file
#
#

# Dependencies
from classify import Classify

# Creating object
classify  = Classify()
# Calling main function
confusionMatrix = classify.main()
# Printing confusion matrix
print(confusionMatrix)
