#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Naive Bayes from scratch 
'''

import numpy as np
from math import sqrt, pi, exp


class NaiveBayes:
    def __init__(self):
        self.X_train = []
        self.y_train = []
        self.separated = None
        self.summarize = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        i = 0
        for row in X_train:
            row.append(y_train[i])
            i += 1

        dataset = X_train
        separated = self.separate_by_class(dataset)
        self.separated = separated

        summarize = self.summarize_by_class(dataset)
        self.summarize = summarize

    def predict(self, X_test):
        predictions = []

        for row in X_test:
            prediction = self.predict_one(self.summarize, row)
            predictions.append(prediction)

        return predictions

    def separate_by_class(self, data):
        separated = {}
        for i in range(len(data)):
            row = data[i]
            class_value = row[-1]
            if class_value not in separated:
                separated[class_value] = []
            separated[class_value].append(row)
        return separated

    def summarize_dataset(self, dataset):
        summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
        del (summaries[-1])
        return summaries

    def summarize_by_class(self, dataset):
        summaries = {}
        for class_value, rows in self.separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    def calculate_probability(self, x, mean, std):
        exponent = exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (sqrt(2 * pi) * std)) * exponent

    def calculate_class_probabilities(self, summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = {}
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, std, count = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, std)
        return probabilities

    def predict_one(self, summaries, row):
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label
