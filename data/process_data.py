import sys
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load Data function
    
    This function loads the data from csv files.
    
    Arguments:
        messages_filepath -> file location for the messages file
        categories_filepath -> file location for the categories file
    
    """
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')
    return df

def clean_data(df):
    """
    Clean Data function
    
    This function cleans the data from csv files.
    
    Arguments:
        df -> merged file with messages & categories data
    
    """
    categories = df['categories'].str.split(';', expand = True)
    row = categories.loc[0:0,:]
    name = []
    for column in row: 
        name.append(row[column][0][:-2])
    categories.columns = name
    
    for column in categories:
    
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast="integer")
    
    df.drop('categories', inplace=True, axis=1)
    df = pd.concat([df, categories], axis =1, join = 'inner')
    df.drop_duplicates(inplace=True)
       
    return df
    
def save_data(df, database_filename):
    """
    Save Data function
    
    This function saves the data from csv files.
    
    Arguments:
        df -> merged & cleaned file with messages & categories data
        database_filename -> file which is saved 
    
    """

    engine = create_engine('sqlite:///' + database_filename)
#     table_name = database_filename.replace(".db","")
    table_name = os.path.basename(database_filename).replace(".db","")
    df.to_sql(table_name, engine, if_exists='replace', index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()