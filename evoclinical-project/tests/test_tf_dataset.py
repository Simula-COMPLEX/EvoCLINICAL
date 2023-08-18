#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/04/2023 21:47
# @Author  : Chengjie
# @File    : test_tf_dataset.py
# @Software: PyCharm


import unittest

import numpy
import tensorflow as tf
import numpy as np


class TesTfDataset(unittest.TestCase):

    def test_dataset(self):
        features, labels = (np.random.sample((4, 2, 1)), np.random.sample((4, 2, 10)))
        print(features)
        print(labels)
        dataset = [features, labels]
        # print(dataset)
        print(tf.ragged.constant(dataset))

        # print(tf.concat([features, labels], axis=2))
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        print(dataset)
