import sys
#!pip install --upgrade setuptools
# !pip install --upgrade pip
# !pip install xgboost

# importing libraries for the ML Process
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
pd.set_option('display.max_column', None)

# import library statements
import re
import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
# import xgboost as xgb

from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")



def load_data(database_filepath):

    """
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","")
    df = pd.read_sql_table(table_name, engine)
    
#     engine = create_engine('sqlite:///data/DisRes.db')
#     table_name = os.path.basename(database_filepath).replace(".db","")
#     df = pd.read_sql_table('DisRes', engine)

    #Remove child alone as it has all zeros only
    # df.drop(['child_alone'], axis=1, inplace = True)
    
    # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 as it is majority class
    # df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    X = df['message']
    y = df.iloc[:,4:]
    
    category_names = y.columns # This will be used for visualization purpose
    return X, y, category_names


def tokenize(text):
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
            
    model = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)), 
                ('tfidf_transformer', TfidfTransformer())
            ])), 
            ('starting_verb_transformer', StartingVerbExtractor())
        ])), 
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Arguments:
        model -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        
    """
    
    params = {'classifier__estimator__max_depth': [2, 3, 4],
              'classifier__estimator__n_estimators': [20, 25, 30]}

    cv = GridSearchCV(model, param_grid=params, n_jobs=-1)
    cv.fit(X_train, y_train)

    y_pred = cv.best_estimator_.predict(X_test)
    
    micro_f1 = f1_score(y_test, y_pred, average = 'micro')
    overall_accuracy = (y_pred == y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    print('F1 score {0:.2f}%'.format(micro_f1*100))

    # Print the whole classification report.
    y_pred = pd.DataFrame(y_pred, columns = y_test.columns)
    
    for column in y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(y_test[column], y_pred[column]))



def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
