# Perceptron

Code to classify dataset using perceptron algorithm

## Editing Code
* Main code is present inside `index.py` file in root directory
* Helper functions are present in `helpers/` directory
* No of features and data size can be changed in `config.json` file in root directory

## Running Code
* Place file containing dataset in root directory
* Run `python index.py` from root directory

## Dataset Model
* Dataset is inside a csv file in root directory
* Each data is represented in a row
* Each feature is present in a cell
* Initial cell in a row represents serial number of data
* Final cell in a row represents class
* Example: `S.No,feature1,feature2,feature3,class`

## Features
* Plot each epoch
* Save plot as a `.png` file
* Calculate no of misclassified points
* Calculate error percentage
