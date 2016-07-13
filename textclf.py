import pandas as pd
from celery import Celery
from svm import train_svm, get_tfidf

app = Celery('textclf', backend='mongodb://127.0.0.1:27017/textclf_backend', broker='redis://127.0.0.1:6379/10')
data = pd.read_csv('data.csv').as_matrix()
clf, vectorizer = train_svm(data)

@app.task
def classificate(tweet):
  tfidf = get_tfidf(vectorizer.transform(tweet))
  pred = clf.predict(tfidf)[0]
  return pred