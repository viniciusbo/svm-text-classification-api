from pymongo import MongoClient
import csv
from random import shuffle

def get_docs():
  client = MongoClient('mongodb://127.0.0.1:27017')
  db = client.tweet_classification
  docs = list(db.dataset1.find({}, { '_id': -1, 't': 1, 'c': 1 }))
  docs = map(parse_doc, docs)
  shuffle(docs)
  return docs

def parse_doc(doc):
  return (doc['t'], int(doc['c']))

def export_to_csv(docs, filename):
  with open(filename + '.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['text', 'label'])
    for doc in docs:
      writer.writerow([doc[0].encode('utf-8'), doc[1]])

if __name__ == '__main__':
  print "Exporting dataset..."
  split_proportion = 0.8
  docs = get_docs()
  export_to_csv(docs, 'data')
  print "Dataset exported"