#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 14/04/2023 19:56
# @Author  : Chengjie
# @File    : preprocess.py
# @Software: PyCharm
import os
import random
import sys
import time

"""add sys path"""

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from datetime import datetime
from systemfiles import data_model
import pandas as pd
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from scipy.spatial import distance


rule_profile = {
    'v01': ['V01', 'V02', 'V04', 'V09', 'V11', 'V13', 'V14', 'V15', 'V16', 'V17',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V29', 'V31',
            'V33', 'V38', 'V40', 'V42', 'V43', 'V44', 'V45', 'V49', 'V08', 'V46'],  # 30 rules

    'v02': ['V01', 'V02', 'V04', 'V09', 'V11', 'V13', 'V14', 'V15', 'V16', 'V17',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V29', 'V31',
            'V33', 'V38', 'V40', 'V42', 'V43', 'V44', 'V45', 'V49', 'V08', 'V46',
            'V50', 'V51', 'V52', 'V53', 'V35'],  # 35 rules

    'v04': ['V01', 'V02', 'V04', 'V09', 'V11', 'V13', 'V14', 'V15', 'V16', 'V17',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V29', 'V31',
            'V33', 'V38', 'V40', 'V42', 'V43', 'V44', 'V45', 'V49', 'V08', 'V46',
            'V50', 'V51', 'V52', 'V53', 'V35', 'V03', 'V12', 'V27', 'V28', 'V34'],  # 40 rules

    'v05': ['V01', 'V02', 'V04', 'V09', 'V11', 'V13', 'V14', 'V15', 'V16', 'V17',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V29', 'V31',
            'V33', 'V38', 'V40', 'V42', 'V43', 'V44', 'V45', 'V49', 'V08', 'V46',
            'V50', 'V51', 'V52', 'V53', 'V35', 'V03', 'V12', 'V27', 'V28', 'V34',
            'V64', 'V69', 'V06', 'V10', 'V30'],  # 45 rules

    'v07': ['V01', 'V02', 'V04', 'V09', 'V11', 'V13', 'V14', 'V15', 'V16', 'V17',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V29', 'V31',
            'V33', 'V38', 'V40', 'V42', 'V43', 'V44', 'V45', 'V49', 'V08', 'V46',
            'V50', 'V51', 'V52', 'V53', 'V35', 'V03', 'V12', 'V27', 'V28', 'V34',
            'V64', 'V69', 'V06', 'V10', 'V30', 'V36', 'V37', 'V41', 'V48', 'V65',
            'V66'],  # 51

    'v10': ['V01', 'V02', 'V04', 'V09', 'V11', 'V13', 'V14', 'V15', 'V16', 'V17',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V29', 'V31',
            'V33', 'V38', 'V40', 'V42', 'V43', 'V44', 'V45', 'V49', 'V08', 'V46',
            'V50', 'V51', 'V52', 'V53', 'V35', 'V03', 'V12', 'V27', 'V28', 'V34',
            'V64', 'V69', 'V06', 'V10', 'V30', 'V36', 'V37', 'V41', 'V48', 'V65',
            'V66', 'V67', 'V39', 'V68', 'V70', 'V73']  # 56
}

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


def pre_process_date_to_unix(dataset, col='activationDate'):
    for row in range(len(dataset[col])):
        dataset.loc[row, col] = int(datetime.timestamp(datetime.strptime(dataset[col][row], '%Y-%m-%dT%H:%M:%S')))


def get_para_body_df(para_body):
    inputs = {'environment': [str(para_body[i]['environment']) for i in range(len(para_body))],
              'activationDate': [str(para_body[i]['activationDate']) for i in range(len(para_body))],
              'cancerType': [str(para_body[i]['cancerType']) for i in range(len(para_body))],
              'cancerTypes': [str(para_body[i]['cancerTypes']) for i in range(len(para_body))]}

    for var in list(data_model.variables_for_training.keys()):
        inputs.update({var: [str(para_body[i]['cancerMessages'][0][var])
                             if para_body[i]['cancerMessages'][0][var] != '' else 'NULL or NONE'
                             for i in range(len(para_body))]})

    df = pd.DataFrame(inputs, index=[i for i in range(len(para_body))])
    return df


def preprocess_para_body(para_body):
    # print(para_body)
    baseline_dataset = pd.read_csv(filepath_or_buffer='../../systemfiles/'
                                                      'training_dataset/training_dataset_balanced_v01_1679996129.csv',
                                   sep=',', index_col=0)

    inputs = {'environment': [str(para_body[i]['environment']) for i in range(len(para_body))],
              'activationDate': [str(para_body[i]['activationDate']) for i in range(len(para_body))],
              'cancerType': [str(para_body[i]['cancerType']) for i in range(len(para_body))],
              'cancerTypes': [str(para_body[i]['cancerTypes']) for i in range(len(para_body))]}

    for var in list(data_model.variables_for_training.keys()):
        # variable_values = []
        # for i in range(len(para_body)):
        #     if para_body[i]['cancerMessages'][0][var] != '':
        #         variable_values.append(str(para_body[i]['cancerMessages'][0][var]))
        #     else:
        #         variable_values.append('NULL or NONE')

        inputs.update({var: [str(para_body[i]['cancerMessages'][0][var])
                             if para_body[i]['cancerMessages'][0][var] != '' else 'NULL or NONE'
                             for i in range(len(para_body))]})

    df = pd.DataFrame(inputs, index=[i for i in range(len(para_body))])

    numerical_variable = df[numerical]
    minMaxScalar = MinMaxScaler()
    minMaxScalar.fit(baseline_dataset[numerical])
    numerical_variable = minMaxScalar.transform(numerical_variable)

    categorical_variable = df[categorical]
    ordinal_encoder = OrdinalEncoder()  # OneHotEncoder(sparse_output=False)
    minMaxScalar = MinMaxScaler()
    minMaxScalar.fit(ordinal_encoder.fit_transform(baseline_dataset[categorical]))
    categorical_variable = minMaxScalar.transform(ordinal_encoder.transform(categorical_variable))

    for date_var in date:
        pre_process_date_to_unix(df, date_var)
        pre_process_date_to_unix(baseline_dataset, date_var)
    date_variable = df[date]
    minMaxScalar = MinMaxScaler()
    minMaxScalar.fit(baseline_dataset[date])
    date_variable = minMaxScalar.transform(date_variable)

    # preprocess string values
    string_variable_values = df[string_variables].astype('string')
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    string_document_embedding = []
    for idx, row in string_variable_values.iterrows():
        # string_document_embedding.append(tf.reshape(embed(row), -1))
        string_document_embedding.append(embed(row))

    X = np.hstack([numerical_variable, categorical_variable, date_variable])

    X2 = np.array(string_document_embedding)

    return X, X2


def preprocess_dataset(dataset, version):
    print('preprocessing dataset...')
    baseline_dataset = pd.read_csv(filepath_or_buffer='../../systemfiles/'
                                                      'training_dataset/training_dataset_balanced_v01_1679996129.csv',
                                   sep=',', index_col=0)

    rule_number = {'v01': 30, 'v02': 35, 'v04': 40, 'v05': 45, 'v07': 51, 'v10': 56}[version]
    #  preprocess numerical variable values
    numerical_variable = dataset[numerical]
    minMaxScalar = MinMaxScaler()
    minMaxScalar.fit(baseline_dataset[numerical])
    numerical_variable = minMaxScalar.transform(numerical_variable)

    categorical_variable = dataset[categorical]
    ordinal_encoder = OrdinalEncoder()  # OneHotEncoder(sparse_output=False)
    minMaxScalar = MinMaxScaler()
    minMaxScalar.fit(ordinal_encoder.fit_transform(baseline_dataset[categorical]))
    categorical_variable = minMaxScalar.transform(ordinal_encoder.transform(categorical_variable))

    for date_var in date:
        pre_process_date_to_unix(dataset, date_var)
        pre_process_date_to_unix(baseline_dataset, date_var)
    date_variable = dataset[date]
    minMaxScalar = MinMaxScaler()
    minMaxScalar.fit(baseline_dataset[date])
    date_variable = minMaxScalar.transform(date_variable)

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


def calculate_coverage(sub_ts):
    # print(sub_ts)
    coverages = []
    sub_ts = tf.transpose(sub_ts)
    for r_i in range(len(sub_ts)):
        y, idx = tf.unique(sub_ts[r_i])
        # print(len(y) / len(sub_ts[r_i]))
        coverages.append(len(y) / 4)
    return coverages, np.array(coverages).mean()


def calculate_weighted_coverage_o_diversity(sub_ts, weights, probability_distribution):
    """
    calculate weighted coverage and output diversity (KL-divergence)
    :param probability_distribution:
    :param sub_ts:
    :param weights:
    :return:
    """
    total = len(sub_ts)
    coverages = []
    diversities = []
    sub_ts = tf.transpose(sub_ts)

    for i in range(len(sub_ts)):
        result_code_r_i = sub_ts[i].numpy()
        weights_r_i = weights[i]
        unique, counts = np.unique(result_code_r_i, return_counts=True)
        count = dict(zip(unique, counts))

        weighted_cov = 0
        for r_c in count.keys():
            weighted_cov += weights_r_i[r_c] * (count[r_c] / total)
        coverages.append(weighted_cov / len(count.keys()))

        calculated_distribution_r_i = [count[0] / total if 0 in result_code_r_i else 0,
                                       count[1] / total if 1 in result_code_r_i else 0,
                                       count[2] / total if 2 in result_code_r_i else 0,
                                       count[3] / total if 3 in result_code_r_i else 0]
        standard_distribution_r_i = probability_distribution[i]
        diversities.append(distance.jensenshannon(calculated_distribution_r_i, standard_distribution_r_i))

    return coverages, diversities, np.array(coverages).mean(), np.array(diversities).mean()


def calculate_output_diversity(sub_ts, probability_distribution):
    """
    calculate weighted coverage and output diversity (KL-divergence)
    :param probability_distribution:
    :param sub_ts:
    :return:
    """
    total = len(sub_ts)
    diversities = []
    sub_ts = tf.transpose(sub_ts)

    for i in range(len(sub_ts)):
        result_code_r_i = sub_ts[i].numpy()
        unique, counts = np.unique(result_code_r_i, return_counts=True)
        count = dict(zip(unique, counts))

        calculated_distribution_r_i = [count[0] / total if 0 in result_code_r_i else 0,
                                       count[1] / total if 1 in result_code_r_i else 0,
                                       count[2] / total if 2 in result_code_r_i else 0,
                                       count[3] / total if 3 in result_code_r_i else 0]
        standard_distribution_r_i = probability_distribution[i]
        # diversities.append(distance.jensenshannon(calculated_distribution_r_i, standard_distribution_r_i))
        diversities.append(((np.array(calculated_distribution_r_i) - np.array(standard_distribution_r_i)) ** 2).mean(axis=0))
    return diversities, np.array(diversities).mean()


def calculate_failed_coverage(pred_results, true_results):
    """
    calculate failed message / total, output diversity
    :param pred_results:
    :param true_results:
    :return:
    """
    pred_results = pred_results.numpy().tolist()
    true_results = true_results.numpy().tolist()
    total = len(true_results)
    count_false_pred = 0
    for i in range(total):
        if pred_results[i] != true_results[i]:
            count_false_pred += 1
    return count_false_pred / total
