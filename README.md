# Disaster Response Pipeline Project

### Instructions:

Overall Description -

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. 
The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

a. Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
b. Build a machine learning pipeline to train the which can classify text message in various categories
c. Run a web app which can show model results in real time

Dependencies - 

Python 3.5+
Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
Natural Language Process Libraries: NLTK
SQLlite Database Libraqries: SQLalchemy
Model Loading and Saving Library: Pickle
Web App and Data Visualization: Flask, Plotly

Execution - 

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/

Acknowledgements -

1. Udacity for providing an amazing Data Science Nanodegree Program
2. Figure Eight for providing the relevant dataset to train the model

Other Details - 

1. ETL Pipeline -
	a. ETL pipeline process use basic concepts for Extracting, Transform and Loading. These concepts are used to load the data, clean it in required manner and save it in SQLite format
	b. The files post ETL Transformation will be used for ML pipeline build

2. ML Pipeline
	a. ML Pipeline build transforms the data by using Tokenization
	b. Leverage Pipeline functionality to implement Vectorization and TF-IDF transformation
	c. Build multiple models and picking best classifier model giving best results
	d. Evaluate and Summarize the outcomes of the model