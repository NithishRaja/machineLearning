#
# Index file
#
#

# Dependencies
from classify import Classify

# Variable to store maximum distance between new mean and previous mean
distance = 10
# Variable to set accuracy
distanceAllowed = 0

# Creating object
classify = Classify()
# Continue to alter mean untill required accuracy is achieved
while distanceAllowed<distance:
    # Calling main function
    distance = classify.main()
    # Calling function to get no of misclassified data
    error = classify.calculateError()
    # Printing no of misclassified data
    print("error: ", error*100/1372)
