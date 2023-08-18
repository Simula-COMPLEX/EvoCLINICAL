#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/04/2023 20:09
# @Author  : Chengjie
# @File    : transfer_learning.py
# @Software: PyCharm
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from ccdt_s_construction.ccdt_s_construction import create_branch, create_model, LossHistory


def load_pretrained_model(model_path='save_model/model.ckpt', trainable=True):
    model = tf.keras.models.load_model(model_path, compile=False)
    model.trainable = trainable
    # model.summary()
    return model


def build_transfer_learning_model(pretrained_model_path='save_model/guri_dt_model_1680968124.h5',
                                  cnn1d_input=22, string_height=22, string_width=512, N_branch=0, train=False):
    pretrained_model = load_pretrained_model(model_path=pretrained_model_path, trainable=train)
    # print(pretrained_model.outputs, len(pretrained_model.outputs))
    inputs = [Input(shape=(cnn1d_input, 1)), Input(shape=(string_height, string_width))]
    previous_output = pretrained_model(inputs)

    for i in range(N_branch):
        new_branch = create_branch(cnn1d_input, string_height, string_width, name='branch_{}'.format(30 + i))
        previous_output.append(new_branch(inputs))

    new_model = Model(inputs=inputs, outputs=previous_output)
    # print(new_model.outputs, len(new_model.outputs))
    # new_model.summary()
    return new_model


def evaluate_transfer_learning_model(X1=None, X2=None, y=None, version_b='v01', version_a='v03', pre_model_path='',
                                     new_model_path='', train=False):
    print('Evaluating transfer learning model...')
    rule_number_b = {'v01': 30, 'v02': 35, 'v04': 40, 'v05': 45, 'v07': 51, 'v10': 56}[version_b]
    rule_number_a = {'v01': 30, 'v02': 35, 'v04': 40, 'v05': 45, 'v07': 51, 'v10': 56}[version_a]

    def convert_y(y_in):
        y_o_dict = {}
        y_w_dict = {}
        loss_w_dict = {}
        for r in range(rule_number_a):
            y_outputs_i = y_in[:, r:r + 1]
            y_o_dict.update({'rule_{}'.format(r): tf.squeeze(y_outputs_i)})
            y_w_dict.update({'rule_{}'.format(r): 1})
            # loss_w_dict.update({'rule_{}'.format(r): calculate_weights(tf.argmax(tf.squeeze(y_outputs_i), axis=-1))})
        return [y_o_dict, y_w_dict, loss_w_dict]

    y = convert_y(y)

    model = build_transfer_learning_model(pretrained_model_path=pre_model_path, N_branch=rule_number_a - rule_number_b,
                                          train=train)
    model.compile(loss=['categorical_crossentropy' for i in range(rule_number_a)], loss_weights=list(y[1].values()),
                  optimizer='adam')
    model.fit([X1, X2], list(y[0].values()),
              verbose=2, epochs=20)

    model.save(new_model_path)


def build_transfer_learning_model2(pretrained_model,
                                   cnn1d_input=22, string_height=22, string_width=512, N_branch=0, train=False):
    pretrained_model.trainable = train
    inputs = [Input(shape=(cnn1d_input, 1)), Input(shape=(string_height, string_width))]
    previous_output = pretrained_model(inputs)

    for i in range(N_branch):
        new_branch = create_branch(cnn1d_input, string_height, string_width, name='branch_{}'.format(30 + i))
        previous_output.append(new_branch(inputs))

    new_model = Model(inputs=inputs, outputs=previous_output)
    # print(new_model.outputs, len(new_model.outputs))
    # new_model.summary()
    return new_model


def evaluate_model(s_X1, s_X2, s_y, r_X1, r_X2, r_y,
                   version_b='v01', version_a='v03', pre_model_path='',
                   new_model_path='', loss_path='', training_type='transfer_learning', epoch=60):
    print('Evaluating {} model...'.format(training_type))
    rule_number_b = {'v01': 30, 'v02': 35, 'v04': 40, 'v05': 45, 'v07': 51, 'v10': 56}[version_b]
    rule_number_a = {'v01': 30, 'v02': 35, 'v04': 40, 'v05': 45, 'v07': 51, 'v10': 56}[version_a]

    def convert_y(y_in, rule_number):
        y_o_dict = {}
        y_w_dict = {}
        loss_w_dict = {}
        for r in range(rule_number):
            y_outputs_i = y_in[:, r:r + 1]
            y_o_dict.update({'rule_{}'.format(r): tf.squeeze(y_outputs_i)})
            y_w_dict.update({'rule_{}'.format(r): 1})
            # loss_w_dict.update({'rule_{}'.format(r): calculate_weights(tf.argmax(tf.squeeze(y_outputs_i), axis=-1))})
        return [y_o_dict, y_w_dict, loss_w_dict]

    s_y = convert_y(s_y, rule_number_b)

    if training_type in ['transfer_learning', 'transfer_learning_baseline']:
        model = load_pretrained_model(pre_model_path, trainable=True)
    else:
        model = create_model(cnn1d_input=22, string_height=22, string_width=512, version=version_b)
    loss = ['categorical_crossentropy' for i in range(rule_number_b)]
    model.compile(loss=loss, loss_weights=list(s_y[1].values()),
                  optimizer='adam')

    history_pre = LossHistory(n_rule=rule_number_b)
    model.fit([s_X1, s_X2], list(s_y[0].values()),
              verbose=2, epochs=epoch, callbacks=[history_pre])

    pd.DataFrame(history_pre.history).to_csv(loss_path + 'fine_tuning_{}.csv'.format(training_type), mode='w',
                                             header=True, index=False)

    r_y = convert_y(r_y, rule_number_a)

    t_model = build_transfer_learning_model2(pretrained_model=model, N_branch=rule_number_a - rule_number_b,
                                             train=False)
    if training_type != 'transfer_learning_baseline':
        t_model.compile(loss=['categorical_crossentropy' for i in range(rule_number_a)],
                        loss_weights=list(r_y[1].values()),
                        optimizer='adam')

        history_after = LossHistory(n_rule=rule_number_a)
        t_model.fit([r_X1, r_X2], list(r_y[0].values()),
                    verbose=2, epochs=epoch, callbacks=[history_after])
        pd.DataFrame(history_after.history).to_csv(loss_path + 'training_new_para_{}.csv'.format(training_type), mode='w',
                                                   header=True, index=False)

    t_model.save(new_model_path)


def transfer_old_branch(s_X1, s_X2, s_y, rule_number_b, version_b, pre_model_path, training_type, name, qu):
    print(name)
    if training_type == 'transfer_learning':
        model = load_pretrained_model(pre_model_path, trainable=True)
    else:
        model = create_model(cnn1d_input=22, string_height=22, string_width=512, version=version_b)
    loss = ['categorical_crossentropy' for i in range(rule_number_b)]
    model.compile(loss=loss, loss_weights=list(s_y[1].values()),
                  optimizer='adam')
    model.fit([s_X1, s_X2], list(s_y[0].values()),
              verbose=2, epochs=2)
    qu.put(model)


def train_new_branch(r_X1, r_X2, r_y, version_b, version_a, rule_number_a, rule_number_b, name, qu):
    print(name)
    model = create_model(cnn1d_input=22, string_height=22, string_width=512,
                         version='{}_{}'.format(version_b, version_a))
    model.compile(loss=['categorical_crossentropy' for i in range(rule_number_a - rule_number_b)],
                  loss_weights=list(r_y[1].values()),
                  optimizer='adam')
    model.fit([r_X1, r_X2], list(r_y[0].values()),
              verbose=2, epochs=2)
    qu.put(model)


if __name__ == '__main__':
    load_pretrained_model(model_path='save_model/TL_model_v02_run_1_selection_1000.tf')