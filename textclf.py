import pandas as pd
from celery import Celery
from svm import clf, tfidf_vectorizer, count_vectorizer

celery = Celery('textclf', backend='amqp', broker='amqp://guest@localhost//')
celery.conf.update(
  CELERY_TASK_SERIALIZER='json',
  CELERY_RESULT_SERIALIZER='json'
)
@celery.task
def classificate(tweet):
  tfidf = count_vectorizer.transform([tweet])
  probabilities = clf.predict_proba(tfidf)[0]
  pred = pick_winner_label(probabilities, threshold=0.5)
  return pred, list(probabilities)

def pick_winner_label(probabilities, threshold):
  winner = (-1, .0)
  for (index, prob) in enumerate(probabilities):
    if prob > threshold and prob > winner[1]:
      winner = (index, prob)
  return winner

if __name__ == "__main__":
  celery.start()