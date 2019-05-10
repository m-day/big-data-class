import numpy as np 
import pandas as pd 
import string
import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk import PorterStemmer as Stemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class NaiveBayes():

    def __init__(self):
        self.vectorizer = TfidfVectorizer(analyzer=self.clean_text)
        self.classifier = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', MultinomialNB())
        ])

    def clean_text(self, text):
        text = text.lower()
        text = ''.join([t for t in text if t not in string.punctuation])
        text = [t for t in text.split() if t not in stopwords.words('english')]

        stemmer = Stemmer()

        text = [stemmer.stem(t) for t in text]
        
        return text    

    def load_data(self, data):
        self.x_train = data[0]
        self.x_test = data[1]
        self.y_train = data[2]
        self.y_test = data[3]

    def fit(self):
        self.classifier.fit(self.x_train, self.y_train)

    def predict(self):
        self.predictions = self.classifier.predict(self.x_test)

    def results(self):
        return classification_report(self.predictions, self.y_test)

    def run_experiment(self, data):
        self.load_data(data)
        self.fit()
        self.predict()
        return self.results()

# Spam Data Set
spam = NaiveBayes()

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

results = spam.run_experiment(train_test_split(df['text'], df['label'], test_size=0.20, random_state= 21))

print(results)

# Airline Sentiment Data Set
airline = NaiveBayes()

df = pd.read_csv('airline.csv', encoding='latin-1')[['airline_sentiment', 'text']]
df.columns = ['label', 'text']

results = airline.run_experiment(train_test_split(df['text'], df['label'], test_size=0.20, random_state= 21))

print(results)

# IMDB movie dataset
imdb = NaiveBayes()

df = pd.read_csv('imdb.csv', encoding='latin-1')[['sentiment', 'review']]
df.columns = ['label', 'text']

results = imdb.run_experiment(train_test_split(df['text'], df['label'], test_size=0.20, random_state= 21))

print(results)

