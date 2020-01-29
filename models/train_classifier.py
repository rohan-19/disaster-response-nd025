import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    INPUT
    database_filepath - path to the sqlite database 
    
    OUTPUT
    X - numpy ndarray with message texts
    Y- numpy ndarray with 36 category values for each message
    category_names - list of 36 category names
    
    This function does the following:
    1. Read the messages data into a pandas dataframe
    2. Store the message texts as X and target category values as Y
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages',engine)
    category_names= ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'child_alone', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']
    X = df.message.values
    Y = df[category_names].values
    return X,Y,category_names

def tokenize(text):
    '''
    INPUT
    text - text string to be cleaned 
    
    OUTPUT
    clean_tokens - list of cleaned tokens
    
    This function does the following:
    1. Convert the text to all lowercase
    2. Tokenize text by spliting into tokens by words
    3. USe the wordnet lemmatizer on each token and add it to a clean token list
    '''
    text=text.lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    INPUT
    NONE 
    
    OUTPUT
    cv - Gridsearchcv model on a pipeline testing different parameters
    
    This function does the following:
    1. Create a pipeline with a countvectorizer and TFIDF transformer followed by a multioutput classifier
    2. Initialize a parameter grid for gridsearchcv
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50)))
    ])


    parameters = {
        'vect__ngram_range': ((1,1),(1, 2)),
        'vect__max_df': (0.5, 0.75),
#        'vect__max_features': (None, 5000,10000),
        'tfidf__use_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1,cv=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - the classifier model for the messages
    X_test - test set of messages
    Y_test - actual category labels for the test messages
    category_names - list of 36 category names 
    
    OUTPUT
    NONE
    
    This function does the following:
    1. Use the classifier model to predict category labels for the test messages
    2. Print the classification report for each of the 36 categories
    '''
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test.T[i], Y_pred.T[i],labels=[0,1]))

def save_model(model, model_filepath):
    '''
    INPUT
    model - the classifier model for the messages
    model_filepath - file path to store the classifier model as pickle
    
    OUTPUT
    NONE
    
    This function does the following:
    1. Store the classifier model as a pickle object at the specified file path
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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