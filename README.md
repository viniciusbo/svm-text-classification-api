# Text Classification REST API

## `dataset.py`

### Data model

```json
{
  "t": <string>,
  "c": <number>
}
```

Running `python dataset.py` will create 3 separate files:

1. `data.csv` contains all data
2. `train_data.csv` contains 80% of the dataset (meant for training)
3. `test_data.csv` contains 20% of the dataset (meant for testing)

## `svm.py`

Running `python svm.py` will train the SVM and print classification report using `train_data.csv` and `test_data.csv`.

## `textclf.py`

Initialize job queue:

```bash
celery -A textclf worker
```

## License

ISC