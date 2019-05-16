#
# Index file
#
#

# Dependencies
from classify import Classify

# Creating object
classify = Classify()
distance = 10
distanceAllowed = 0
while distanceAllowed<distance:
    distance = classify.main()
