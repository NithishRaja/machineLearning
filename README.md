# Perceptron

Code to perform naive bayes analysis

## Editing Code
* Main code is present inside `index.py` file in root directory
* Helper functions are present in `helpers/` directory
* No of features and data size can be changed in `config.json` file in root directory

## Running Code
* Place file containing dataset in `dataset` directory
* Update data file name in `config.json` file
* Run `python index.py` from root directory

## Dataset Model
* Dataset is inside a txt file
* Each data instance is represented in a line
* Initial word represents category of data
* Second word indicates the class of data
* Third word gives name of data file
* Remaining words in line are the data
* Example: `music neg 544.txt i was misled and thought i was buying the entire cd and it contains one song `

## Features
* Can classify more than 2 classes
* Calculate frequency of each word and persist it to a file
* Perform 80-20 data split for training and validation of data
* Calculate probability of a data instance belonging to class
* Calculate confusion matrix
