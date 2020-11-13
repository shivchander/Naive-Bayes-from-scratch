#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
Naive Bayes from scratch using Iris Dataset
'''
from NaiveBayes import NaiveBayes
from utils import *


def main():
    x, y, = load_dataset()

    acc_dict = {}
    f1_beta_half_dict = {}
    f1_beta_1_dict = {}
    f1_beta_2_dict = {}
    roc_dict = {}
    for bins in [5, 10, 15, 20]:
        skf = StratifiedKFold(n_splits=bins)
        accs = []
        f1_beta_half = []
        f1_beta_1 = []
        f1_beta_2 = []
        for train_index, test_index in skf.split(x, y):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            nb = NaiveBayes()
            X_train, X_test = X_train.tolist(), X_test.tolist()
            y_train, y_test = y_train.ravel().tolist(), y_test.ravel().tolist()
            nb.fit(X_train, y_train)
            y_hat = nb.predict(X_test)
            accs.append(balanced_accuracy_score(y_test, y_hat))
            f1_beta_half.append(fbeta_score(y_test, y_hat, beta=0.5))
            f1_beta_1.append(fbeta_score(y_test, y_hat, beta=1))
            f1_beta_2.append(fbeta_score(y_test, y_hat, beta=2))
        acc_dict[bins] = accs
        roc_dict[bins] = get_fpr_tpr(y_test, y_hat)
        f1_beta_half_dict[bins] = f1_beta_half
        f1_beta_1_dict[bins] = f1_beta_1
        f1_beta_2_dict[bins] = f1_beta_2

    print(print_accuracies(acc_dict))
    plot_accuracies(acc_dict, 'Naive Bayes Accuracy')
    plot_f1(f1_beta_half_dict, 'Naive Bayes F1 Score (Beta = 0.5)')
    plot_f1(f1_beta_1_dict, 'Naive Bayes F1 Score (Beta = 1)')
    plot_f1(f1_beta_2_dict, 'Naive Bayes F1 Score (Beta = 2)')
    plot_roc(roc_dict, 'Naive Bayes ROC Curve')


if __name__ == '__main__':
    main()
