# Getting Started with the Sentiment Analyser Tool


## Getting Started
In order to setup the application we need to run the API and the UI. Information on getting setup on these exist within the README.md in both the API and UI directory.

## Information

The Classifier directory consists of Python files for the development of the proccessed dataset and the files for creating TFIDF, SVD and classifier models.
This tool uses a classifier to analyse text and tweets and retrieve their predicted sentiment. 


The Data directory consists of the raw data used for training the data along with any subsequent files which was outputted from the preprocessing for the classifier. 


The Diagrams directory consists of analysis of different models to visualise their efficiency.


The Models directory consists of the resulting TFIDF, SVD and classifier models produced from the classifier directory.

## How to obtain models from raw data

1) Uncompress the compressed data folder
2) Run Classifier/data_cleaning.py - Which cleans the data and produces a new cleaned file
3) Run Classifier/data_preprocessiing.py - Which preprocessed the data to be ready to train the model
4) Run Classifier/model_building.py - Which produces the required models and outputs it to the Models directory

The Classifier/model_exploration.py file was used to test different types of models and their efficiency.


## Notes
1) TwitterAPI keys are required in the relevant file
2) A sample of the dataset is provided
