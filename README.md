# K - Means Classifier

Code to classify dataset using k means classifier algorithm

## Editing Code
* Main code is present inside `index.py` file in root directory
* Helper functions are present in `helpers/` directory
* No of features and data size can be changed in `config.json` file in root directory

## Running Code
* Place file containing dataset in `dataset` directory
* Update name of file containing dataset in `config.json` file
* Run `python index.py` from root directory

## Dataset Model
* Dataset is inside a csv file in dataset directory
* Each data is represented in a row
* Each feature is present in a cell
* Final cell in a row represents class of data
* Example: `feature1,feature2,feature3,class`

## Features
* Gives no of misclassified points
* Similarity measures can be easily changed
