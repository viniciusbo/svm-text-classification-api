# import numpy
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from unidecode import unidecode
import re

def extract_features(docs):
  docs = map(preprocess_text, docs)
  hashvector, vectorizer = hashing_vectorizer(docs)
  tfidf = get_tfidf(hashvector)
  return tfidf, vectorizer

def preprocess_text(text):
  text = text.lower()
  text = unidecode(text)
  text = remove_rt(text)
  text = remove_twitter_user_mentions(text)
  text = remove_hashtags(text)
  text = remove_links(text)
  # text = remove_numbers(text)
  return text

def remove_rt(text):
  return re.sub('RT ', '', text)

def remove_twitter_user_mentions(text):
  return re.sub(r'(?:@[\w_]+)', '', text)

def remove_hashtags(text):
  return re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', text)

def remove_links(text):
  return re.sub(r'http\S+', '', text)

def remove_numbers(text):
  return re.sub(r'?:(?:\d+,?)+(?:\.?\d+)?)', '', text)

def hashing_vectorizer(corpus):
  vectorizer = HashingVectorizer(stop_words=nltk.corpus.stopwords.words('portuguese'), strip_accents='ascii')
  counts = vectorizer.transform(corpus)
  return counts, vectorizer

def get_tfidf(counts):
  transformer = TfidfTransformer()
  tfidf = transformer.fit_transform(counts)
  return tfidf

def train_svm(data):
  X, vectorizer = extract_features(row[0] for row in data)
  x = [row[1] for row in data]
  clf = svm.LinearSVC()
  clf.fit(X, x)
  return clf, vectorizer

def build_report(clf, vectorizer, test_data):
  y_true = [row[1] for row in test_data]
  y_pred = [clf.predict(get_tfidf(vectorizer.transform([row[0]])))[0] for row in test_data]
  report = classification_report(y_true, y_pred)
  return report

if __name__ == '__main__':
  # numpy.set_printoptions(threshold='nan')
  print 'Loading data...'
  data = pd.read_csv('train_data.csv', encoding='utf8').as_matrix()
  print 'Loaded {} data'.format(len(data))
  print 'Finished data loading'

  print 'Training SVM...'
  clf, vectorizer = train_svm(data)
  print 'SVM trained'

  print 'Building report...'
  test_data = pd.read_csv('test_data.csv', encoding='utf8').as_matrix()
  report = build_report(clf, vectorizer, test_data)
  print report