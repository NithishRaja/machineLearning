#
# Index file
#
#

# Dependencies
import os
from parse import Parse

parse = Parse("naive_bayes_data.txt", "data.json")
parse.getData()
parse.writeFrequencyData()
