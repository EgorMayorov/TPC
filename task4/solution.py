import pickle
import json
import numpy as np
import scipy


class Solution:
    def __init__(self):
        with open('model.pkl', "rb") as file:
            self.model = pickle.load(file)
        with open('transformer.pkl', "rb") as file:
            self.transformer = pickle.load(file)
        with open('dev-dataset-task2023-04.json') as f:
            raw_data = json.load(f)
        X, y = [], []
        for data, label in raw_data:
            X.append(data)
            y.append(label)
        X = np.array(X)
        self.y = np.array(y)
        self.X_tfidf = self.transformer.transform(X)

    def predict(self, text: str) -> str:
        data = self.transformer.transform([text])
        pred = self.model.predict(data)
        self.X_tfidf = scipy.sparse.vstack([self.X_tfidf, data])
        self.y = np.concatenate([self.y, pred])
        self.model.fit(self.X_tfidf, self.y)
        return str(pred[0])
