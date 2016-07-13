import pandas as pd
from celery import Celery
from svm import train_svm, tfidf_vectorizer

celery = Celery('textclf', backend='amqp', broker='amqp://guest@localhost//')
celery.conf.update(
  CELERY_TASK_SERIALIZER='json',
  CELERY_RESULT_SERIALIZER='json'
)
data = pd.read_csv('data.csv', encoding='utf8').as_matrix()
clf = train_svm(data)

@celery.task
def classificate(tweet):
  tfidf = tfidf_vectorizer.transform(tweet)
  probabilities = clf.predict_proba(tfidf)[0]
  pred = pick_winner_label(probabilities, threshold=0.5)
  return list(probabilities), pred

def pick_winner_label(probabilities, threshold):
  winner = (-1, .0)
  for (index, prob) in enumerate(probabilities):
    if prob > threshold and prob > winner[1]:
      winner = (index, prob)
  return winner

if __name__ == "__main__":
  celery.start()