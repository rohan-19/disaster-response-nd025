# Disaster Response Pipeline Project

## Summary
The aim of the project is to create a web application for each classification of disaster related messages posted online during any calamity. 

With millions of messages being posted every second, it is very difficult to go through each one to try to understand the context. The application helps you figure out the top themes talked about in each message based on a classifier model trained on historical data.


## File Descriptions:
1. data - The folder contains two csvs one with the messages data and other with their associated categories. The process_data.py file takes in both the datasets as input and stores the merged and cleaned dataset to a sqlite database at the specified path.

2. models - The folder contains a train_classifier.py file which takes in path to the sqlite database containing the cleaned up messages data and trains a classifier model on it. The final model is stored as a pickle object at the specified path.

3. app - The folder contains a flask application in run.py which serves the static web app files from the templates included. 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
