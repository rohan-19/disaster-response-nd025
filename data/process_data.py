import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - path to the messages csv
    categories_filepath - path to the categories csv
    
    OUTPUT
    df - pandas dataframe with messages and their categories merged together
    
    This function does the following:
    1. Read in the messages and categories datasets from csvs
    2. Merge the two datasets on id and return the resulting dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id',how='left')
    return df

def clean_data(df):
    '''
    INPUT
    df - the messages dataframe to be cleaned 
    
    OUTPUT
    df - pandas dataframe cleaned messages and categories
    
    This function does the following:
    1. Split the categories column into columns for each distinct category
    2. Add the category names as column headers for the categories datafram
    3. Change each value to a 1 or 0 in the categories dataframe by removing name prefix
    4. Change type of category dataframe columns to int
    5. Append the categories to the original messages dataframe
    '''

    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2].strip())
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
      categories[column] = categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
      categories[column] = categories[column].astype('int32')

    df=df.drop(['categories'],axis=1)

    df = pd.concat([df,categories],axis=1)
    df=df.drop_duplicates()

    return df

def save_data(df, database_filename):
    '''
    INPUT
    df - pandas dataframe to be written to SQLite
    database_filename - file path for the SQLite database
    
    OUTPUT
    NONE
    
    This function does the following:
    1. Create sqlite engine at the specified file path
    2. Store the dataframe in the sqlite database in messages table
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)  


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