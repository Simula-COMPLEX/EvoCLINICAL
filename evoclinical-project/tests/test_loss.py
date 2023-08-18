#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16/03/2023 21:54
# @Author  : Chengjie
# @File    : test_loss.py
# @Software: PyCharm
import unittest

import tensorflow as tf
import numpy as np
from torch import nn
import torch


class TestLoss(unittest.TestCase):

    def test_softmax_cross_entropy_with_logits(self):
        y_logits = tf.constant([[[100, 0, 0], [0, 11, -10]],
                                [[-11, 0, 8], [10, 8, -2]]], dtype='float32')

        y_true = tf.constant([[[1, 0, 0], [0, 1, 0]],
                              [[0, 0, 1], [1, 0, 0]]], dtype='float32')

        # print(tf.nn.softmax(y_logits).numpy())

        for i in range(y_true.shape[0]):
            print(tf.nn.softmax_cross_entropy_with_logits(y_true[i], y_logits[i], axis=-1))

        print(tf.nn.softmax_cross_entropy_with_logits(y_true, y_logits, axis=-1))
        #
        # print(tf.compat.v1.losses.softmax_cross_entropy(y_true, y_logits))

    def test_onehot_inverse(self):
        x = numpy.array([[0, 1, 2, 3, 0], [1, 2, 3, 0, 1]])
        x_onehot = tf.one_hot(x, 4, axis=-1)

        print(x_onehot)
        print(tf.argmax(x_onehot, axis=-1))

    def test_tf_unicode_trans(self):
        # a = tf.strings.unicode_transcode(["Hello", "TensorFlow", "2.x"], "Integer", "UTF-16-BE")
        print(ord("1"))

    def test_cross_en(self):
        loss = nn.CrossEntropyLoss()
        y_t = torch.from_numpy(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])).float()
        y_p = torch.from_numpy(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])).float()

        out_put = loss(y_t, y_p)
        # out_put.backward()

        print(out_put.item())

        print(torch.randn(3, 5, requires_grad=True))

        input = torch.rand(64, 70, 5)
        target = torch.empty(64, 5, dtype=torch.long).random_(5)
        print(target)

        loss = nn.CrossEntropyLoss()
        output = loss(input, target)

        print('input: ', input, input.shape)
        print('target: ', target, target.shape)
        print('Cross Entropy Loss: ', output)

    def test_weighted_loss(self):
        # np.random.seed(123)

        # let's say we have the logits and labels of a batch of size 6 with 5 classes
        logits = tf.constant(np.random.randint(0, 10, 30).reshape(6, 5), dtype=tf.float32)
        # labels = tf.constant(np.random.randint(0, 5, 6), dtype=tf.int32)

        labels = tf.constant(np.random.randint(0, 1, 30).reshape(6, 5), dtype=tf.int32)
        print(logits)
        print(tf.argmax(logits, axis=-1))
        print(labels)

        # specify some class weightings
        class_weights = tf.constant([[0.3, 0.1, 0.2, 0.3, 0.1, 0.1], [0.3, 0.1, 0.2, 0.3, 0.1, 0.1],
                                     [0.3, 0.1, 0.2, 0.3, 0.1, 0.1], [0.3, 0.1, 0.2, 0.3, 0.1, 0.1],
                                     [0.3, 0.1, 0.2, 0.3, 0.1, 0.1]])

        # class_weights = tf.constant([0.3, 0.1, 0.2, 0.3, 0.1])

        # specify the weights for each sample in the batch (without having to compute the onehot label matrix)
        weights = tf.gather(class_weights, labels)
        #
        print(weights)

        # compute the loss
        result = tf.compat.v1.losses.softmax_cross_entropy(labels, logits, class_weights)
        print(result)

    def test_softmax_cross_entropy(self):
        y_true = [0, 0, 1]
        y_pred = [0.1, 0.8, 0.1]
        weights = [0.3, 0.7, 0]
        smoothing = 0.2
        loss = tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred, weights=weights,
                                                         label_smoothing=smoothing).numpy()
        print(loss)

    def test_weighted_loss_function(self):
        def calculating_class_weights(y_true):
            from sklearn.utils import compute_class_weight
            number_dim = np.shape(y_true)[1]
            weights = np.empty([number_dim, 4])
            for i in range(number_dim):
                weights[i] = compute_class_weight(class_weight='balanced', classes=np.unique(y_true), y=y_true[:, i])
            return weights

        y_t = np.array([[1, 0, 0, 0],
                        [3, 0, 1, 2],
                        [2, 0, 0, 1],
                        [3, 1, 2, 1],
                        [0, 2, 0, 3]])
        print(y_t)
        weight = calculating_class_weights(y_t)
        print(weight)
        print(weight[:, 0] ** y_t)
