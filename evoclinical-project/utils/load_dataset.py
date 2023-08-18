#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28/02/2023 16:18
# @Author  : Chengjie
# @File    : load_dataset.py
# @Software: PyCharm
import random
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import make_multilabel_classification
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, OrdinalEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow_hub as hub

print(mpl.get_backend())
# mpl.use('TkAgg')


def label_barchart(f_n='../systemfiles/training_dataset_reordered4.csv'):
    dataset = pd.read_csv(filepath_or_buffer=f_n, sep=',')

    # headers = ['V02', 'V03', 'V04', 'V05', 'V09', 'V10', 'V12', 'V15', 'V16', 'V18', 'V29', 'V31', 'V32', 'V33',
    #            'V34', 'V36', 'V37', 'V38', 'V39', 'V41', 'V44', 'V45', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58',
    #            'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V68', 'V69', 'V70', 'V73']

    # y = dataset[headers].values.astype('float32') print(y[0])
    y = dataset.values[:, -70:].astype('float32')
    # tmp_y = []

    label_results = {}
    all_fail_list = []
    for i in range(len(y)):
        y[i] = list(map(lambda x: 0 if x in [2] else 1, y[i]))
        # tmp_y.append(y[i])
        count = list(y[i]).count(0)
        label_results.update({str(count): label_results[str(count)] + 1}) if \
            str(count) in label_results.keys() else label_results.update({str(count): 1})

    print(label_results)
    fig = plt.figure(figsize=(10, 5))
    # creating the bar plot
    plt.bar(list(label_results.keys()), list(label_results.values()), color='maroon',
            width=0.4)
    plt.xlabel("Number of rules that failed")
    plt.ylabel("Number of messages")
    plt.title("")
    plt.show()


def pre_process_date_to_unix(dataset, col='activationDate'):
    for row in range(len(dataset[col])):
        dataset.loc[row, col] = int(datetime.timestamp(datetime.strptime(dataset[col][row], '%Y-%m-%dT%H:%M:%S')))


def load_data(f_n='../systemfiles/training_dataset_reordered6.csv', version='v01', balance=False):
    print('loading dataset: {}'.format(f_n))
    dataset = pd.read_csv(filepath_or_buffer=f_n, sep=',', index_col=0)  # , nrows=1000
    # dataset = dataset.sample(n=200)
    # dataset.reset_index(drop=True, inplace=True)

    print('total number of parameter bodies: {}'.format(len(dataset)))

    rule_number = {'v01': 30, 'v70': 70}[version]
    rule_results = dataset.values[:, -rule_number:].astype('float32')

    if not balance:
        all_2_array = []
        for i in range(len(rule_results)):
            if list(rule_results[i]).count(2) == rule_number:
                all_2_array.append(i)
        print(len(all_2_array))
        # print(len(list(set(all_2_array + no_1_array))))
        drop = np.random.choice(all_2_array, 7000, replace=False).astype(int)
        dataset = dataset.drop(drop, axis=0).reset_index(drop=True)
        # dataset = dataset.reset_index(drop=True, inplace=True)
        print('number of parameter bodies after balancing: {}'.format(len(dataset)))

        dataset.to_csv('../systemfiles/training_dataset/training_dataset_reordered_sampled_{}_{}.csv'.
                       format(version, str(int(time.time()))), mode='w', index=True, header=True)

    print(dataset)

    # return

    categorical = ['environment', 'cancerType', 'cancerTypes', 'metastase', 'meldingstype', 'kjonn', 'annenBehandling']
    numerical = ['ds', 'stralebehandling', 'kjemoterapi', 'hormonbehandling',
                 'kirurgi', 'topografi', 'lokalisasjon', 'ekstralokalisasjon',
                 'morfologi', 'basis'
                 ]

    date = ['activationDate', 'meldingsdato', 'diagnosedato', 'fodselsdato', 'dodsdato']

    string_variables = [
        'kirurgi', 'topografi', 'lokalisasjon', 'ekstralokalisasjon', 'morfologi', 'basis',

        'multiplisitet', 'pt', 'ct', 'ypt', 'cn', 'pn', 'ypn', 'cm', 'pm', 'ypm',
        'patologiStadium', 'kliniskStadium', 'side', 'sykehuskode', 'topografiICDO3',
        'morfologiICDO3']

    #  preprocess numerical variable values
    numerical_variable = dataset[numerical]
    minMaxScalar = MinMaxScaler()
    numerical_variable = minMaxScalar.fit_transform(numerical_variable)
    # print(numerical_variable, numerical_variable.shape)

    # preprocess categorical variable values
    categorical_variable = dataset[categorical]
    ordinal_encoder = OrdinalEncoder()  # OneHotEncoder(sparse_output=False)
    categorical_variable = MinMaxScaler().fit_transform(ordinal_encoder.fit_transform(categorical_variable))
    # print(categorical_variable, categorical_variable.shape)

    # preprocess date values
    for date_var in date:
        pre_process_date_to_unix(dataset, date_var)
    date_variable = dataset[date]
    date_variable = minMaxScalar.fit_transform(date_variable)
    # print(date_variable)

    # preprocess string values
    string_variable_values = dataset[string_variables].astype('string')
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    string_document_embedding = []
    for idx, row in string_variable_values.iterrows():
        # string_document_embedding.append(tf.reshape(embed(row), -1))
        string_document_embedding.append(embed(row))

    X = np.hstack([numerical_variable, categorical_variable, date_variable])

    X2 = np.array(string_document_embedding)

    # y = dataset[rules].values.astype('float32')
    y = dataset.values[:, -rule_number:].astype('float32') - 1
    y_tf_onehot = tf.one_hot(y, depth=4)

    # Multiclass and multi-output classification
    # https://www.kaggle.com/code/depture/multiclass-and-multi-output-classification

    tf.random.shuffle(X, seed=8)
    tf.random.shuffle(X2, seed=8)
    tf.random.shuffle(y_tf_onehot, seed=8)

    print('Input: {}'.format(X.shape))
    print('Output: {}'.format(y_tf_onehot.shape))

    # print(tf.gather(y_tf_onehot, [1, 2, 3]))

    return X, X2, y_tf_onehot


def get_dataset():
    X, y = make_multilabel_classification(n_samples=20, n_features=10, n_classes=6, n_labels=10, random_state=1)
    return X, y


def over_sampling(f_n='../systemfiles/training_dataset_reordered6.csv', version='v70'):
    print('loading dataset: {}'.format(f_n))
    dataset = pd.read_csv(filepath_or_buffer=f_n, sep=',', index_col=0)

    print('total number of parameter bodies: {}'.format(len(dataset)))

    rule_number = {'v01': 30, 'v70': 70}[version]
    rule_results = dataset.values[:, -rule_number:].astype('float32')

    validation_rules = pd.read_excel('../systemfiles/rules/xlsx/Validering_rule_{}.xlsx'.format(version),
                                     sheet_name='Validering rule')
    validation_rule_id = list(validation_rules['Rule Number'])
    rule_headers = [r_id for r_id in validation_rule_id if r_id == r_id]
    if 'V07' in rule_headers:
        rule_headers.remove('V07')

    for r in ['V01', 'V03', 'V04', 'V06', 'V08', 'V10', 'V15', 'V13', 'V14', 'V16',
              'V17', 'V19', 'V20', 'V21',
              'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V34', 'V35',
              'V36', 'V37', 'V38', 'V40', 'V41', 'V42', 'V43', 'V46', 'V47', 'V48',
              'V50', 'V51', 'V52', 'V64', 'V65', 'V67']:
        rule_headers.remove(r)

    # rule_headers = ['V12', 'V15', 'V18', 'V27', 'V30', 'V38', 'V43', 'V49']

    file_name = '../systemfiles/dataset_{}/training_dataset_reordered_sampled.csv'.format(version)
    pd.DataFrame([dataset.columns.values]).to_csv(file_name, mode='w', index=True, header=False)

    def sample(data, ru, result_c):
        df = data[data[ru] == result_c].astype('string')
        if len(df) == 0:
            return data.astype('string')
        elif 1000 >= len(df) > 0:
            df_new = pd.DataFrame(np.repeat(df.values, int(1000 / len(df)), axis=0)).astype('string')
            df_new.columns = df.columns
            data = pd.concat([data, df_new], axis=0).astype('string')
            data.reset_index(drop=True)
            # pass
        # elif len(df) > 5000:
        #     drop = np.random.choice(df.index, len(df) - 5000, replace=False).astype(int)
        #     data = data.drop(drop, axis=0).reset_index(drop=True).astype('string')
        return data

    for rule in rule_headers:
        for r_c in ['1']:
            print(rule + '_' + str(r_c))
            dataset = sample(dataset, rule, r_c)
            # dataset.columns = h

    dataset.astype('string').to_csv(file_name, mode='a', index=True, header=False)


def down_sample(f_n='', version='v70'):
    # dataset = pd.read_csv(filepath_or_buffer=f_n, sep=',')
    #
    # dataset.to_csv(f_n, mode='w', index=True, header=True)

    dataset = pd.read_csv(filepath_or_buffer=f_n, sep=',', index_col=0)
    print(dataset)

    rule_results = dataset.values[:, -70:].astype('float32')
    all_2_array = []
    for i in range(len(rule_results)):
        if list(rule_results[i]).count(2) == 70:
            all_2_array.append(i)
    print(len(all_2_array))
    # print(len(list(set(all_2_array + no_1_array))))
    drop = np.random.choice(all_2_array, round(len(all_2_array) * 9 / 10), replace=False).astype(int)
    dataset = dataset.drop(drop, axis=0).reset_index(drop=True)

    validation_rules = pd.read_excel('../systemfiles/rules/xlsx/Validering_rule_{}.xlsx'.format(version),
                                     sheet_name='Validering rule')
    validation_rule_id = list(validation_rules['Rule Number'])
    rule_headers = [r_id for r_id in validation_rule_id if r_id == r_id]
    if 'V07' in rule_headers:
        rule_headers.remove('V07')

    def down_sampling(data, ru, result_c):
        drop_list = []
        for i in range(len(data)):
            if data[ru][i] == result_c:
                drop_list.append(i)

        if len(drop_list) > 10000:
            dr = np.random.choice(drop_list, len(drop_list) - 10000, replace=False).astype(int)
            data = data.drop(dr, axis=0).reset_index(drop=True)
        return data

    # for rule in rule_headers:
    #     print(rule)
    #     for r_c in [2, 3, 4]:
    #         print(r_c)
    #         d_s = down_sampling(dataset, rule, r_c)
    #         dataset = d_s
    dataset.to_csv('../systemfiles/dataset_{}/training_dataset_reordered_sampled.csv'.format(version),
                   mode='w', header=True, index=False)


if __name__ == '__main__':
    # input_x1, input_x2, output_y = load_data(f_n='../systemfiles/dataset_v70/training_dataset_reordered_sampled.csv',
    #                                          version='v70', balance=False)

    over_sampling(f_n='../systemfiles/training_dataset/training_dataset_balanced_1679778619.csv', version='v70')

    # down_sample(f_n='../systemfiles/dataset_v70/training_dataset_sampled.csv', version='v70')
