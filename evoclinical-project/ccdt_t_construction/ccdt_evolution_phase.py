#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 09/04/2023 17:25
# @Author  : Chengjie
# @File    : ccdt_evolution_phase.py
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
from utils.preprocess import preprocess_para_body, preprocess_dataset
from transfer_learning import load_pretrained_model
import tensorflow as tf
import pandas as pd
from systemfiles.data_model import variables_for_training
from transfer_learning import evaluate_model
from ccdt_s_construction.ccdt_s_construction import evaluate_pretraining_model
import argparse
from utils.preprocess import rule_profile

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

CANCER_DATASET = pd.read_csv(filepath_or_buffer='../../systemfiles/parameter_bodies.csv', sep=',')


def get_rule_results(test_results, version='v01'):
    validation_rule_id = rule_profile[version]

    result_codes = []

    for j in range(len(validation_rule_id)):
        for result in test_results['results'][0]['messageValidationRuleResults']:
            if result['ruleNumber'] == validation_rule_id[j]:
                result_codes.append(result['resultCode'] - 1)
                break

    return result_codes


def get_training_data(test_results, version):
    print('Preprocessing training data...')
    title = ['environment', 'activationDate', 'cancerType', 'cancerTypes']
    title = title + list(variables_for_training.keys())

    validation_rule_id = rule_profile[version]

    title = title + list(validation_rule_id)

    training_data = []
    for i in range(len(test_results)):
        # parameter_body = json.loads(test_results['Parameter Body'][i].replace('\'', '\"'))
        if str(test_results['Test Result'][i]) == 'exception':
            continue
        try:
            parameter_body = eval(test_results['Parameter Body'][i])
            test_result = eval(test_results['Test Result'][i])
        except TypeError:
            parameter_body = test_results['Parameter Body'][i]
            test_result = test_results['Test Result'][i]

        if 'cancerMessages' not in parameter_body.keys():
            continue

        if test_result['preAggregationResult'] is None:
            model_input = [str(parameter_body['environment']), str(parameter_body['activationDate']),
                           str(parameter_body['cancerType']), str(parameter_body['cancerTypes'])]
            for key in variables_for_training.keys():
                model_input.append(str(parameter_body['cancerMessages'][0][key]) if
                                   parameter_body['cancerMessages'][0][key] != '' else 'NULL or NONE')

            for j in range(len(validation_rule_id)):
                for result in test_result['results'][0]['messageValidationRuleResults']:
                    if result['ruleNumber'] == validation_rule_id[j]:
                        model_input.append(result['resultCode'])
                        break
            training_data.append(model_input)
    # print(training_data)
    df = pd.DataFrame(training_data, columns=title)

    return preprocess_dataset(df, version)


def load_test_suites(transfer_version, RUN, ID, version_b, version_a, model_path, search_algorithm='IBEA', epoch=60):
    file_id = random.randint(0, 99)
    # file_id = 27
    directory = '/global/D1/homes/qinghua/guri/test_suites_v2/'
    # directory = '/Users/chengjielu/Work/FSE2023/TestGURI/rulevalidationpredict/multi_outputs_model/test_suites_v2/'
    file_n = directory + 'Run_{}/{}/{}/Selection_{}/test_suites_{}.csv'.format(RUN, transfer_version,
                                                                               search_algorithm,
                                                                               ID,
                                                                               file_id)

    print('Loading test suites, file {}...'.format(file_n))
    test_suites = pd.read_csv(filepath_or_buffer=file_n)
    subset_length = len(test_suites)

    f = open(
        directory + 'Run_{}/{}/{}/Selection_{}/reference_file_name'.format(RUN, transfer_version,
                                                                           search_algorithm,
                                                                           ID) + '.txt',
        'w', encoding='utf-8')
    f.writelines('test_suites_{}.csv'.format(file_id))
    f.close()

    # test_suites['Comparison Result'] = test_suites['Parameter Body']

    def balance_upsampling(ts):
        failed_list = []
        passed_list = []
        for i in range(len(ts['Test Result'])):
            result_codes = get_rule_results(test_results=eval(ts['Test Result'][i]), version=version_b)
            try:
                pred = eval(ts['Predictions'][i])
            except TypeError:
                pred = ts['Predictions'][i]

            if pred != result_codes:
                failed_list.append(i)
            else:
                passed_list.append(i)

        failed_ts = ts.loc[failed_list]
        passed_ts = ts.loc[passed_list]

        failed_len = len(failed_ts)
        pass_len = len(passed_ts)

        print('original size: failed, passed', failed_len, pass_len)

        if failed_len < pass_len:
            failed_ts = failed_ts.loc[failed_ts.index.repeat(int(pass_len / failed_len))].reset_index(drop=True)
            failed_len = len(failed_ts)
            if 1 < pass_len / failed_len < 2:
                failed_ts = pd.concat([failed_ts, failed_ts.sample(pass_len - failed_len)])
        elif failed_len > pass_len:
            passed_ts = passed_ts.loc[passed_ts.index.repeat(int(failed_len / pass_len))].reset_index(drop=True)
            pass_len = len(passed_ts)
            if 1 < failed_len / pass_len < 2:
                passed_ts = pd.concat([passed_ts, passed_ts.sample(failed_len - pass_len)])

        print('failed, passed length', len(failed_ts), len(passed_ts))
        print('===========================================')
        ts = pd.concat([failed_ts, passed_ts])
        ts.reset_index(drop=True, inplace=True)
        if len(ts) < 5000:
            # print(len(ts), int(5000 / len(ts)))
            ts = ts.loc[ts.index.repeat(int(5000 / len(ts)))].reset_index(drop=True)
            # print(len(ts))
            if 1 < 5000 / len(ts) < 2:
                # print(len(ts))
                ts = pd.concat([ts, ts.sample(5000 - len(ts))])
        ts.reset_index(drop=True, inplace=True)
        return ts

    print('Getting failed test suites...')
    transfer_test_suites = balance_upsampling(test_suites)

    pretrain_test_suites = CANCER_DATASET.sample(5000)
    pretrain_test_suites.reset_index(drop=True, inplace=True)

    pretrain_test_suites_transfer = pd.concat([pretrain_test_suites, transfer_test_suites])
    pretrain_test_suites_transfer.reset_index(drop=True, inplace=True)

    # create training dataset for training from stretch
    baseline_test_suite = CANCER_DATASET.sample(subset_length)
    baseline_test_suite.reset_index(drop=True, inplace=True)

    pre_model = load_pretrained_model(model_path)
    pred_softmax = pre_model.predict(
        [preprocess_para_body([eval(x) for x in baseline_test_suite['Parameter Body'].values])])
    pred_results = tf.transpose(tf.squeeze(tf.argmax(pred_softmax, axis=-1))).numpy()
    # print(pred_results)

    baseline_test_suite['Predictions'] = pred_results.tolist()

    print('Getting failed baseline test suites...')
    base_transfer_test_suites = balance_upsampling(baseline_test_suite)

    pretrain_test_suites_baseline = pd.concat([pretrain_test_suites, base_transfer_test_suites])
    pretrain_test_suites_baseline.reset_index(drop=True, inplace=True)

    X1_b, X2_b, y_b = get_training_data(transfer_test_suites, version_b)
    X1_b_base, X2_b_base, y_b_base = get_training_data(base_transfer_test_suites, version_b)

    X1_a, X2_a, y_a = get_training_data(pretrain_test_suites_transfer, version_a)
    X1_a_base, X2_a_base, y_a_base = get_training_data(pretrain_test_suites_baseline, version_a)

    tl_s = time.time()
    evaluate_model(X1_b, X2_b, y_b, X1_a, X2_a, y_a,
                   version_b=version_b, version_a=version_a,
                   pre_model_path=model_path,
                   new_model_path=directory + 'Run_{}/{}/{}/Selection_{}/TL_model_{}_run_{}_selection_{}.tf'.
                   format(RUN, transfer_version, search_algorithm, ID, version_a, RUN, ID),
                   loss_path=directory + 'Run_{}/{}/{}/Selection_{}/'.format(RUN, transfer_version,
                                                                             search_algorithm, ID),
                   training_type='transfer_learning', epoch=epoch)
    tl_e = time.time()

    """
    baseline for active transfer learning: using random dataset to do fine-tuning
    """
    tl_b_s = time.time()
    evaluate_model(X1_b_base, X2_b_base, y_b_base, X1_a, X2_a, y_a,
                   version_b=version_b, version_a=version_a,
                   pre_model_path=model_path,
                   new_model_path=directory + 'Run_{}/{}/{}/Selection_{}/TLB_model_{}_run_{}_selection_{}.tf'.
                   format(RUN, transfer_version, search_algorithm, ID, version_a, RUN, ID),
                   loss_path=directory + 'Run_{}/{}/{}/Selection_{}/'.format(RUN, transfer_version,
                                                                             search_algorithm, ID),
                   training_type='transfer_learning_baseline', epoch=epoch)
    tl_b_e = time.time()

    tfs_s = time.time()
    evaluate_model(X1_b, X2_b, y_b, X1_a_base, X2_a_base, y_a_base,
                   version_b=version_b, version_a=version_a,
                   pre_model_path=None,
                   new_model_path=directory + 'Run_{}/{}/{}/Selection_{}/TFS_model_{}_run_{}_selection_{}.tf'.
                   format(RUN, transfer_version, search_algorithm, ID, version_a, RUN, ID),
                   loss_path=directory + 'Run_{}/{}/{}/Selection_{}/'.format(RUN, transfer_version,
                                                                             search_algorithm, ID),
                   training_type='training_from_stretch', epoch=epoch)
    tfs_e = time.time()

    f = open(
        directory + 'Run_{}/{}/{}/Selection_{}/model_training_time'.format(RUN,
                                                                           transfer_version,
                                                                           search_algorithm,
                                                                           ID) + '.txt', 'w',
        encoding='utf-8')
    f.writelines(str(tl_e - tl_s) + '\n')
    f.writelines(str(tl_b_e - tl_b_s) + '\n')
    f.writelines(str(tfs_e - tfs_s))
    f.close()


def pretrain_model(version):
    # training_dataset = CANCER_DATASET.sample(n=35000)
    # training_dataset.reset_index(drop=True, inplace=True)
    training_dataset = pd.read_csv(filepath_or_buffer='./save_model/training_dataset/training_dataset.csv', sep=',')

    X1, X2, y = get_training_data(test_results=training_dataset, version=version)
    evaluate_pretraining_model(X1, X2, y, version=version,
                               new_model_path='./save_model/guri_dt_model_{}.tf'.format(version),
                               epoch=60)


if __name__ == '__main__':
    """
        v01: 30 rules
        v02: 35 rules

        v04: 40 rules
        v05: 45 rules

        v07: 51 rules
        v10: 56 rules
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", type=str, default='v01_v02', help="vsource_vtarget")
    parser.add_argument("--run", type=int, default=1, help="index of run, starting from 1")
    parser.add_argument("--size", type=int, default=1000, help="test suite size")
    parser.add_argument("--evaluation", type=int, default=20000)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--search_algorithm", type=str, default='IBEA')
    args = parser.parse_args()

    for t_v in [args.transfer]:
        print('============================================')
        print('Output directory: /global/D1/homes/qinghua/guri/test_suites_v2/'
              'Run_{}/{}/{}/Selection_{}'.format(args.run, t_v, args.search_algorithm, args.size))
        print('============================================')
        version_before = t_v.split('_')[0]
        version_after = t_v.split('_')[1]

        load_test_suites(transfer_version=t_v, RUN=args.run, ID=args.size, version_b=version_before,
                         version_a=version_after, model_path='./save_model/guri_dt_model_{}.tf'.format(version_before),
                         search_algorithm=args.search_algorithm,
                         epoch=args.epoch)
