#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 03/03/2023 11:25
# @Author  : Chengjie
# @File    : metrics.py
# @Software: PyCharm
import json

import tensorflow as tf


def recall(y_true, y_pred):
    recall_list = []
    recall_m = tf.keras.metrics.Recall()
    recall_m.update_state(y_true, y_pred)
    recall_list.append(format(recall_m.result().numpy(), '.3f'))
    recall_m.reset_state()

    y_true = tf.transpose(y_true)
    y_pred = tf.transpose(y_pred)

    for i in range(y_true.shape[0]):
        recall_m.update_state(y_true[i], y_pred[i])
        recall_list.append(format(recall_m.result().numpy(), '.3f'))
        recall_m.reset_state()

    return recall_list


def precision(y_true, y_pred):
    precision_list = []
    precision_m = tf.keras.metrics.Precision()
    precision_m.update_state(y_true, y_pred)
    precision_list.append(format(precision_m.result().numpy(), '.3f'))
    precision_m.reset_state()

    y_true = tf.transpose(y_true)
    y_pred = tf.transpose(y_pred)

    for i in range(y_true.shape[0]):
        precision_m.update_state(y_true[i], y_pred[i])
        precision_list.append(format(precision_m.result().numpy(), '.3f'))
        precision_m.reset_state()

    return precision_list


def accuracy_for_each_rule(y_true, y_pred):
    acc_list = []
    accuracy = tf.keras.metrics.Accuracy()

    accuracy.update_state(y_true, y_pred)
    acc_list.append(format(accuracy.result().numpy(), '.3f'))
    accuracy.reset_state()

    y_true = tf.transpose(y_true)
    y_pred = tf.transpose(y_pred)

    for i in range(y_true.shape[0]):
        accuracy.update_state(y_true[i], y_pred[i])
        acc_list.append(format(accuracy.result().numpy(), '.3f'))
        accuracy.reset_state()
    return acc_list


def preprocess(y, result_c):
    y = y.numpy()
    for i in range(len(y)):
        if y[i] != result_c:
            y[i] = 0
        else:
            y[i] = 1
    return y


def recall_precision_for_each_status_code(y_true, y_pred, status_code):
    y_true = tf.reshape(tf.cast(tf.transpose(tf.argmax(y_true, axis=-1)), tf.float32), [-1])
    y_pred = tf.reshape(tf.cast(tf.transpose(tf.argmax(y_pred, axis=-1)), tf.float32), [-1])

    y_true = preprocess(y_true, status_code)
    y_pred = preprocess(y_pred, status_code)
    recall_m = tf.keras.metrics.Recall()
    recall_m.update_state(y_true, y_pred)

    precision_m = tf.keras.metrics.Precision()
    precision_m.update_state(y_true, y_pred)
    return json.dumps({'Recall_ResultCode({})'.format(status_code): format(recall_m.result().numpy(), '.3f'),
                       'Precision_ResultCode({})'.format(status_code): format(precision_m.result().numpy(), '.3f'),
                       'F1_score_ResultCode({})'.format(status_code):
                           format(2 * (recall_m.result().numpy() * precision_m.result().numpy()) /
                                  (recall_m.result().numpy() + precision_m.result().numpy()), '.3f')})


def get_precision_metric(result_code):

    def precision_result(y_true, y_pred):
        y_true = tf.reshape(tf.cast(tf.transpose(tf.argmax(y_true, axis=-1)), tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(tf.transpose(tf.argmax(y_pred, axis=-1)), tf.float32), [-1])
        y_true = preprocess(y_true, result_code)
        y_pred = preprocess(y_pred, result_code)
        precision_m = tf.keras.metrics.Precision()
        precision_m.update_state(y_true, y_pred)
        return tf.cast(precision_m.result(), tf.float32)

    precision_result.__name__ = 'precision_result_{}'.format(result_code)
    return precision_result


def get_precision_metric_rule(result_code, rule_id):
    def precision_result_for_rule(y_true, y_pred):
        # y_true_r_id = tf.cast(tf.transpose(tf.argmax(y_true, axis=-1)), tf.float32)[rule_id]
        # y_pred_r_id = tf.cast(tf.transpose(tf.argmax(y_pred, axis=-1)), tf.float32)[rule_id]

        y_true_r_id = tf.argmax(y_true, axis=-1)
        y_pred_r_id = tf.argmax(y_pred, axis=-1)

        y_true_r_id = preprocess(y_true_r_id, result_code)
        y_pred_r_id = preprocess(y_pred_r_id, result_code)
        precision_m = tf.keras.metrics.Precision()
        precision_m.update_state(y_true_r_id, y_pred_r_id)
        return tf.cast(precision_m.result(), tf.float32)
    precision_result_for_rule.__name__ = 'precision_rule{}_resultCode{}'.format(rule_id, result_code)
    return precision_result_for_rule


def get_recall_metric_rule(result_code, rule_id):
    def recall_result_for_rule(y_true, y_pred):
        # y_true_r_id = tf.cast(tf.transpose(tf.argmax(y_true, axis=-1)), tf.float32)[rule_id]
        # y_pred_r_id = tf.cast(tf.transpose(tf.argmax(y_pred, axis=-1)), tf.float32)[rule_id]

        y_true_r_id = tf.argmax(y_true, axis=-1)
        y_pred_r_id = tf.argmax(y_pred, axis=-1)

        y_true_r_id = preprocess(y_true_r_id, result_code)
        y_pred_r_id = preprocess(y_pred_r_id, result_code)
        recall_m = tf.keras.metrics.Recall()
        recall_m.update_state(y_true_r_id, y_pred_r_id)
        return tf.cast(recall_m.result(), tf.float32)
    recall_result_for_rule.__name__ = 'recall_rule{}_resultCode{}'.format(rule_id, result_code)
    return recall_result_for_rule


def get_recall_metric(result_code):

    def recall_result(y_true, y_pred):
        y_true = tf.reshape(tf.cast(tf.transpose(tf.argmax(y_true, axis=-1)), tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(tf.transpose(tf.argmax(y_pred, axis=-1)), tf.float32), [-1])
        y_true = preprocess(y_true, result_code)
        y_pred = preprocess(y_pred, result_code)
        recall_m = tf.keras.metrics.Recall()
        recall_m.update_state(y_true, y_pred)
        return tf.cast(recall_m.result(), tf.float32)

    recall_result.__name__ = 'recall_result_{}'.format(result_code)
    return recall_result


def get_f1_score_metric(result_code):
    precision_metric = get_precision_metric(result_code)
    recall_metric = get_recall_metric(result_code)

    def f1_score_result(y_true, y_pred):
        pre = precision_metric(y_true, y_pred)
        rec = recall_metric(y_true, y_pred)
        return tf.cast(2 * (pre * rec) / (pre + rec), tf.float32)

    f1_score_result.__name__ = 'f1_score_result_{}'.format(result_code)
    return f1_score_result


# def f1_score(y_true, y_pred):
#     # f1_list = []
#     f1 = tfa.metrics.F1Score(num_classes=len(y_true[0]))
#     f1.update_state(y_true, y_pred)
#     return list(f1.result().numpy())


class MultiLabel_Accuracy(tf.keras.metrics.Accuracy):
    # https: // blog.csdn.net / weixin_39122088 / article / details / 106694214
    def __int__(self, name='', *args, **kwargs):
        super(MultiLabel_Accuracy, self).__init__(name=name, *args, **kwargs)
        self.rule_id = int(self.name.split('-')[1])
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        rule_id = int(self.name.split('_')[1])
        y_true = tf.cast(tf.transpose(tf.argmax(y_true, axis=-1)), tf.float32)
        y_pred = tf.cast(tf.transpose(tf.argmax(y_pred, axis=-1)), tf.float32)

        values = tf.cast(tf.math.equal(y_true[rule_id], y_pred[rule_id]), tf.float32)
        # print(y_true[0], y_true[0].shape, tf.shape(y_true[0])[0])
        self.total.assign_add(tf.cast(tf.shape(y_true[rule_id])[0], tf.float32))
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total

    def reset_state(self):
        self.total.assign(0)
        self.count.assign(0)


if __name__ == '__main__':
    y_t = tf.convert_to_tensor([[[10, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]],
                               dtype=tf.float32)  # , [1, 0, 1, 0], [0, 0, 1, 0]
    y_p = tf.convert_to_tensor([[[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]],
                               dtype=tf.float32)  # , [1, 0, 1, 0], [0, 0, 1, 0]

    f1_s = get_f1_score_metric(0)

    print(get_precision_metric(0).__name__)
