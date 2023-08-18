#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 27/02/2023 14:39
# @Author  : Chengjie
# @File    : test_load.py
# @Software: PyCharm
import time
import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder


class TestLoad(unittest.TestCase):

    def test_load_data(self):
        dataset = pd.read_csv(filepath_or_buffer='../../systemfiles/dataset_v70/previous/training_dataset.csv', sep=',')
        data = dataset.values

        X = data[:, :-97].astype(str)
        Y = data[:, :-70].astype(str)

        # ordinal = OrdinalEncoder()
        # X = ordinal.fit_transform(X)

        print('Input: {}'.format(X.shape))
        print(X[0])

        print('Output: {}'.format(Y.shape))
        print(Y[0])

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=int(time.time()))

        # ordinal encode input variables
        ordinal_encoder = OrdinalEncoder()
        ordinal_encoder.fit(X)
        X_train = ordinal_encoder.transform(X_train)
        X_test = ordinal_encoder.transform(X_test)

        # define the model
        model = LogisticRegression(solver='liblinear', max_iter=500)
        # fit on the training set
        model.fit(X_train, y_train)
        # predict on test set
        yhat = model.predict(X_test)
        # evaluate predictions

        print(yhat)
        accuracy = accuracy_score(y_test, yhat)
        print('Accuracy: %.2f' % (accuracy * 100))

    def test_encoder(self):
        dataset = pd.read_csv(filepath_or_buffer='../../systemfiles/dataset_v70/previous/training_dataset.csv', sep=',')
        data = dataset.values
        print(data[0])
        print(dataset[['environment', 'cancerType']])
        # test OneHotEncoder
        onehot_encoder = OneHotEncoder()
        x = onehot_encoder.fit_transform(np.reshape(dataset[['environment', 'cancerType']].values, (-1, 2)))

        # test OneHotEncoder
        ordinal_encoder = OrdinalEncoder()
        x = ordinal_encoder.fit_transform(np.reshape(dataset[['environment', 'cancerType']].values, (-1, 2)))

        # test LabelEncoder
        label_encoder = LabelEncoder()
        x = label_encoder.fit_transform(dataset[['cancerType']])

        print(x)


