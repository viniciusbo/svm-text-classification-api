# import numpy
import nltk
import pandas as pd
from sklearn import svm, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from unidecode import unidecode
import re

tfidf_vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('portuguese'), strip_accents='ascii')
count_vectorizer = CountVectorizer(analyzer='word', stop_words=nltk.corpus.stopwords.words('portuguese'), strip_accents='ascii')

def train_svm(data):
  X = extract_features(row[0] for row in data)
  x = [row[1] for row in data]
  # clf = svm.LinearSVC()
  clf = svm.SVC(kernel='linear', probability=True)
  clf.fit(X, x)
  return clf

def extract_features(docs):
  docs = map(preprocess_text, docs)
  features = count_vectorizer.fit_transform(docs)
  return features

def preprocess_text(text):
  text = text.lower()
  text = unidecode(text)
  text = remove_rt(text)
  text = remove_twitter_user_mentions(text)
  text = remove_hashtags(text)
  text = remove_links(text)
  text = remove_special_chars(text)
  text = remove_numbers(text)
  # TODO:
  # - remove "kkkkkk"
  return text

def remove_rt(text):
  return re.sub('RT ', '', text)

def remove_twitter_user_mentions(text):
  return re.sub(r'(?:@[\w_]+)', '', text)

def remove_hashtags(text):
  return re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', text)

def remove_links(text):
  return re.sub(r'http\S+', '', text)

def remove_special_chars(text):
  return re.sub(r'[^\w\s\']', '', text)

def remove_numbers(text):
  return re.sub(r'\d', '', text)

def build_classification_report(clf, test_data):
  y_true = [row[1] for row in test_data]
  docs = map(preprocess_text, [row[0] for row in test_data])
  tfidf = count_vectorizer.transform(docs)
  y_pred = clf.predict(tfidf)
  report = classification_report(y_true, y_pred)
  return report

def cross_validation_report(clf, dataset):
  data = count_vectorizer.transform([row[0] for row in dataset])
  target = [row[1] for row in dataset]
  return cross_validation.cross_val_score(clf, data, target)

data = pd.read_csv('data.csv', encoding='utf8').as_matrix()
clf = train_svm(data)

if __name__ == '__main__':
  # numpy.set_printoptions(threshold='nan')
  print 'Loading data...'
  data = pd.read_csv('train_data.csv', encoding='utf8').as_matrix()
  print 'Loaded {} data'.format(len(data))
  print 'Finished data loading'

  print 'Training SVM...'
  clf = train_svm(data)
  print 'SVM trained'

  print 'Building reports...'
  print 'Classification report:'
  test_data = pd.read_csv('test_data.csv', encoding='utf8').as_matrix()
  print build_classification_report(clf, test_data)
  print '----------'
  print 'Cross-validation report:'
  dataset = pd.read_csv('data.csv', encoding='utf8').as_matrix()
  print cross_validation_report(clf, dataset)