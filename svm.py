# import numpy
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer, CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

def extract_features(docs):
  hashvector, vectorizer = hashing_vectorizer(docs)
  tfidf = get_tfidf(hashvector)
  return tfidf, vectorizer

def hashing_vectorizer(corpus):
  vectorizer = HashingVectorizer(stop_words=nltk.corpus.stopwords.words('portuguese'), strip_accents='ascii')
  counts = vectorizer.transform(corpus)
  return counts, vectorizer

def get_tfidf(counts):
  transformer = TfidfTransformer()
  tfidf = transformer.fit_transform(counts)
  return tfidf

# numpy.set_printoptions(threshold='nan')
print 'Loading data...'
data = pd.read_csv('train_data.csv').as_matrix()
print 'Loaded {} data'.format(len(data))
print 'Finished data loading'

print 'Training SVM...'
X, vectorizer = extract_features(row[0] for row in data)
x = [row[1] for row in data]
clf = svm.LinearSVC()
clf.fit(X, x)
print 'SVM trained'

print 'Building report...'
test_data = pd.read_csv('test_data.csv').as_matrix()
y_true = [row[1] for row in test_data]
y_pred = [clf.predict(get_tfidf(vectorizer.transform([row[0]])))[0] for row in test_data]
report = classification_report(y_true, y_pred)
print report