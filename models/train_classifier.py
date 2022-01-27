import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download(['wordnet', 'punkt'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster', engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = y.columns.values
    
    return X, y, category_names


def tokenize(text):
    # tokenize sententce into words
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize the words, normalize to lower case and strip the spaces
    clean_tokens = []
    
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens


class CountChars(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        chars_num = pd.Series(X).apply(lambda x: len(x)).values
        return pd.DataFrame(chars_num)
    

def build_model():
    # use the tuned model with the new feature
    rf_pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipe', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('count_chars', CountChars())])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return rf_pipeline


def evaluate_model(model, X_test, y_test, category_names):
    # make predictions 
    y_pred = model.predict(X_test)
    
    # iterate through the columns and generate reports
    for col in range(36):
        pred = y_pred[:, col]
        actual = y_test.iloc[:,col]
        name = category_names[col]

        # print the results for each col
        print(f'======================Feature {name}======================')
        print(classification_report(actual, pred))


def tune_model(model, X_train, y_train):
    # use grid search for best params
    parameters = {
        'clf__estimator__n_estimators': [10, 50],
        'clf__estimator__class_weight': [None, 'balanced']
    }

    cv = GridSearchCV(model, parameters, cv=5)
    cv.fit(X_train, y_train)

    print(f'Best Params: {cv.best_params_}')
    
    return cv
    
        
def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))

    
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
        
        print('Tuning model...')
        tuned_model = tune_model(model, X_train, Y_train)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(tuned_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()