# Disaster Response Pipeline Project

### Objective and Solution:

This project focus on building an ETL & ML driven pipeline process to classify key tweets/messages from users stuck in disaster situations to classify these messages into certain categories associated with type of help required. The process will help us classify the messages on realtime basis to enhance disaster response from respective authorities/departments. The application developed focus on providing an interface for end-users to leverage this model to classify such messages on realtime basis 

### Overall Description -

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis. This project is divided in the following key sections:

a. Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
b. Build a machine learning pipeline to train the which can classify text message in various categories
c. Run a web app which can show model results in real time

### Dependencies - 

* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly

### Execution - 

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/

### Acknowledgements -

1. Udacity for providing an amazing Data Science Nanodegree Program
2. Figure Eight for providing the relevant dataset to train the model

### Other Details - 

1. ETL Pipeline -
	a. ETL pipeline process use basic concepts for Extracting, Transform and Loading. These concepts are used to load the data, clean it in required manner and save it in SQLite format
	b. The files post ETL Transformation will be used for ML pipeline build

2. ML Pipeline
	a. ML Pipeline build transforms the data by using Tokenization
	b. Leverage Pipeline functionality to implement Vectorization and TF-IDF transformation
	c. Build multiple models and picking best classifier model giving best results
	d. Evaluate and Summarize the outcomes of the model
	e. GridSearch and adding FeatureUnion to refine the model output
	
3. Setting up the Application and visuals
	a. Setup charts and link the databases to update the application
	
### Important Files
**app/templates/***: templates/html files for web app

**data/process_data.py**: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

**data/ETL Pipeline Preparation.ipynb**: Jupyter notebook with elaborate analysis and EDA required for building refined ETL pipeline

**data/disaster_pipeline.csv and data/disaster_categories.csv**: Input data for the messages and categories associated with the messages

**models/train_classifier.py**: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

**models/ML Pipeline Preparation.ipynb**: Jupyter notebook with elaborate analysis and EDA required for building refined ML pipeline with GridSearch and FeatureUnion

**run.py**: This file can be used to launch the Flask web app used to classify disaster messages
