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
classify.main()
classify.calculateError()
