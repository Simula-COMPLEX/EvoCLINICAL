#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 06/04/2023 22:46
# @Author  : Chengjie
# @File    : ccdt_s_construction.py
# @Software: PyCharm
import time

import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Reshape, Input
from sklearn.model_selection import RepeatedKFold
from utils.metrics import recall_precision_for_each_status_code
from utils.load_dataset import load_data


def calculate_weights(y):
    count = {'0': 0.01, '1': 0.01, '2': 0.01, '3': 0.01}
    for i in range(len(y)):
        if y[i] == 0:
            count['0'] += 1
        elif y[i] == 1:
            count['1'] += 1
        elif y[i] == 2:
            count['2'] += 1
        elif y[i] == 3:
            count['3'] += 1
    weights = [(1 / count['0'] * (len(y) / 2)), (1 / count['1'] * (len(y) / 2)),
               (1 / count['2'] * (len(y) / 2)), (1 / count['3'] * (len(y) / 2))]
    return weights


def create_cnn1d(height, width, name):
    model = Sequential(name=name)
    model.add(Conv1D(16, 3, input_shape=(height, width), activation='relu'))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    # model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    # if input_type == 'string':
    #     model.add(Conv1D(128, 3, activation='relu'))
    # model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    # model.add(Dense(200, activation='relu'))
    return model


def create_branch(cnn1d_input, string_height, string_width, name='branch_0'):
    cnn_number_ca = create_cnn1d(cnn1d_input, 1, name='{}_cnn_number'.format(name))
    cnn_string = create_cnn1d(string_height, string_width, name='{}_cnn_string'.format(name))
    combined_input = tf.keras.layers.concatenate([cnn_number_ca.output, cnn_string.output])

    x = Dense(200, activation='relu')(combined_input)
    # x = Dense(200, activation='relu')(x)
    x = Dense(4, activation='softmax')(x)

    branch = Model(inputs=[cnn_number_ca.input, cnn_string.input], outputs=x, name=name)
    branch.trainable = True
    return branch


def create_model(cnn1d_input, string_height, string_width, version='v01'):
    rule_number = {'v01': 30, 'v02': 35, 'v04': 40, 'v05': 45, 'v07': 51, 'v10': 56}[version]
    outputs = []
    inputs = [Input(shape=(cnn1d_input, 1)), Input(shape=(string_height, string_width))]
    for i in range(rule_number):
        branch_i = create_branch(cnn1d_input, string_height, string_width, 'branch_{}'.format(i))
        outputs.append(branch_i(inputs))

    guri_dt = Model(inputs=inputs, outputs=outputs, name='guri_dt')
    guri_dt.summary()
    return guri_dt


def evaluate_model(X1=None, X2=None, y=None, version='v01'):
    def convert_y(y_in):
        y_o_dict = {}
        y_w_dict = {}
        loss_w_dict = {}
        for r in range(y.shape[1]):
            y_outputs_i = y_in[:, r:r + 1]
            y_o_dict.update({'rule_{}'.format(r): tf.squeeze(y_outputs_i)})
            y_w_dict.update({'rule_{}'.format(r): 1})
            loss_w_dict.update({'rule_{}'.format(r): calculate_weights(tf.argmax(tf.squeeze(y_outputs_i), axis=-1))})
        return [y_o_dict, y_w_dict, loss_w_dict]

    x1_input, x2_input_length, x2_input_width = X1.shape[1], X2.shape[1], X2.shape[2]
    cv = RepeatedKFold(n_splits=7, n_repeats=1, random_state=1)
    for train_ix, test_ix in cv.split(X1):
        X1_train, X1_test = X1[train_ix], X1[test_ix]
        X2_train, X2_test = tf.gather(X2, train_ix), tf.gather(X2, test_ix)
        y_train, y_test = tf.gather(y, train_ix), tf.gather(y, test_ix)
        y_train_dict, y_test_dict = convert_y(y_train), convert_y(y_test)

        # print(y_train_dict[0]['rule_0'])

        model = create_model(cnn1d_input=x1_input, string_height=x2_input_length, string_width=x2_input_width,
                             version=version)
        # loss = [tf.nn.softmax_cross_entropy_with_logits for i in range(30)]
        loss = ['categorical_crossentropy' for i in range(y.shape[1])]
        # loss = [weighted_categorical_crossentropy(y_train_dict[2]['rule_{}'.format(i)]) for i in range(30)]
        model.compile(loss=loss, loss_weights=list(y_train_dict[1].values()),
                      optimizer=tf.optimizers.Adam(), )

        model.fit([X1_train, X2_train], list(y_train_dict[0].values()),
                  verbose=2, epochs=60,
                  # callbacks=tf_callback
                  )
        model.save('./save_model/guri_dt_model_{}.h5'.format(str(int(time.time()))))

        yhat_logits = model.predict([X1_test, X2_test])
        yhat_t_logits = model.predict([X1_train, X2_train])

        print(len(yhat_logits))
        print(tf.argmax(list(y_test_dict[0].values()), axis=-1))
        print(tf.argmax(yhat_logits, axis=-1))

        for status_code in [0, 1, 2, 3]:
            print(recall_precision_for_each_status_code(list(y_test_dict[0].values()), yhat_logits, status_code))
            print(recall_precision_for_each_status_code(list(y_train_dict[0].values()), yhat_t_logits, status_code))

        break


class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, n_rule):
        super().__init__()
        self.history = None
        self.number_of_rules = n_rule

    def on_train_begin(self, logs={}):
        # self.history = {'batch_loss': [], 'epoch_loss': []}
        self.history = {'batch_loss': []}

        for i in range(self.number_of_rules):
            self.history.update({'branch_{}_batch_loss'.format(i): [],
                                 # 'branch_{}_epoch_loss'.format(i): []
                                 })
        self.history.update({'timestep': []})

    def on_batch_end(self, batch, logs={}):
        self.history['batch_loss'].append(logs.get('loss'))

        for i in range(self.number_of_rules):
            self.history['branch_{}_batch_loss'.format(i)].append(logs.get('branch_{}_loss'.format(i)))
        self.history['timestep'].append(time.time())

    # def on_epoch_end(self, epoch, logs={}):
    #     self.history['epoch_loss'].append(logs.get('loss'))
    #
    #     for i in range(self.number_of_rules):
    #         self.history['branch_{}_epoch_loss'.format(i)].append(logs.get('branch_{}_loss'.format(i)))


def evaluate_pretraining_model(X1=None, X2=None, y=None, version='v01', new_model_path='', epoch=60):
    print('Evaluating pretraining learning model...')
    rule_number = {'v01': 30, 'v02': 35, 'v04': 40, 'v05': 45, 'v07': 51, 'v10': 56}[version]

    def convert_y(y_in):
        y_o_dict = {}
        y_w_dict = {}
        loss_w_dict = {}
        for r in range(y.shape[1]):
            y_outputs_i = y_in[:, r:r + 1]
            y_o_dict.update({'rule_{}'.format(r): tf.squeeze(y_outputs_i)})
            y_w_dict.update({'rule_{}'.format(r): 1})
            loss_w_dict.update({'rule_{}'.format(r): calculate_weights(tf.argmax(tf.squeeze(y_outputs_i), axis=-1))})
        return [y_o_dict, y_w_dict, loss_w_dict]

    y_convert = convert_y(y)

    x1_input, x2_input_length, x2_input_width = X1.shape[1], X2.shape[1], X2.shape[2]
    model = create_model(cnn1d_input=x1_input, string_height=x2_input_length, string_width=x2_input_width,
                         version=version)
    loss = ['categorical_crossentropy' for i in range(y.shape[1])]
    model.compile(loss=loss, loss_weights=list(y_convert[1].values()),
                  optimizer='adam', )
    history = LossHistory(n_rule=rule_number)

    r = model.fit([X1, X2], list(y_convert[0].values()),
                  verbose=2, epochs=epoch,
                  callbacks=[history]
                  )

    pd.DataFrame(history.history).to_csv('./test.csv', mode='w', header=True, index=False)
    for key in history.history.keys():
        # print(history.history['batch_loss'], len(history.history['batch_loss']))
        print(key)
        print(history.history[key], len(history.history[key]))
    # model.save(new_model_path)


if __name__ == '__main__':
    ver = 'v70'
    X1_inputs, X2_inputs, y_outputs = load_data(f_n='/Users/chengjielu/Work/FSE2023/TestGURI'
                                                    '/systemfiles/training_dataset'
                                                    '/training_dataset_balanced_1679778619.csv'.
                                                format(ver),
                                                version=ver, balance=True)

    evaluate_model(X1_inputs, X2_inputs, y_outputs, version=ver)
