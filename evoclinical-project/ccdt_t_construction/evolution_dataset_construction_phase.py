#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 09/04/2023 17:25
# @Author  : Chengjie
# @File    : evolution_dataset_construction_phase.py
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
import requests
from jmetal.algorithm.multiobjective import NSGAII, IBEA
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

from utils.preprocess import preprocess_para_body, calculate_coverage, \
    preprocess_dataset, rule_profile
from search_based_selection.problem import TestSuiteSelection
from transfer_learning import load_pretrained_model
import numpy as np
import tensorflow as tf
import pandas as pd
from systemfiles.data_model import variables_for_training
from ccdt_s_construction.ccdt_s_construction import evaluate_pretraining_model
import argparse

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

CANCER_DATASET = pd.read_csv(filepath_or_buffer='../../systemfiles/dataset_v70/parameter_bodies.csv', sep=',')
RESULT_CODE_WEIGHTS = pd.read_csv(filepath_or_buffer='../../systemfiles/TS/weights.csv', sep=',')
PROB_DISTRIBUTIONS = pd.read_csv(filepath_or_buffer='../../systemfiles/TS/probability_distribution.csv', sep=',')


def search_based_selection(model, test_suite, RUN, ID, transfer_version='v01_v03', search_algorithm='NSGA-II',
                           evaluation=15000):
    print('Search-based Test Suite Minimization...')
    ts_size = len(test_suite)

    ver_before = transfer_version.split('_')[0]

    # directory = '/Users/chengjielu/Work/FSE2023/TestGURI/rulevalidationpredict/multi_outputs_model/test_suites_v2/'
    directory = '/global/D1/homes/qinghua/guri/test_suites_v2/'
    if not os.path.exists(
            directory + 'Run_{}/{}/{}/Selection_{}'.format(RUN, transfer_version, search_algorithm, ID)):
        os.makedirs(
            directory + 'Run_{}/{}/{}/Selection_{}'.format(RUN, transfer_version, search_algorithm, ID))

    problem = TestSuiteSelection(test_suite=test_suite, guri_dt=model,
                                 result_code_weights=np.array(RESULT_CODE_WEIGHTS[rule_profile[ver_before]].values).T,
                                 probability_distribution=np.array(
                                     PROB_DISTRIBUTIONS[rule_profile[ver_before]].values).T,
                                 number_of_bits=ts_size, run=RUN, ID=ID,
                                 search_algorithm=search_algorithm,
                                 transfer_version=transfer_version)
    max_evaluations = evaluation

    if search_algorithm == 'NSGA-II':
        algorithm = NSGAII(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            mutation=BitFlipMutation(probability=1.0 / ts_size),
            crossover=SPXCrossover(probability=0.9),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )

    elif search_algorithm == 'IBEA':
        algorithm = IBEA(
            problem=problem,
            kappa=1.,
            population_size=100,
            offspring_population_size=100,
            mutation=BitFlipMutation(probability=1.0 / ts_size),
            crossover=SPXCrossover(probability=0.9),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )
    else:
        print('No search algorithm matched, abort.')
        return

    algorithm.run()
    fronts = algorithm.get_result()

    f = open(
        directory + 'Run_{}/{}/{}/Selection_{}/algorithm_computing_time'.format(RUN, transfer_version,
                                                                                search_algorithm,
                                                                                ID) + '.txt',
        'w', encoding='utf-8')
    f.writelines(str(algorithm.total_computing_time))
    f.close()

    for i in range(len(fronts)):
        front = fronts[i]
        true_index = [i for i in range(len(front.variables[0])) if front.variables[0][i] is True]

        # execution_outputs_new_ver = test_suite_execution([test_suite[i] for i in true_index])

        ts_sub = pd.DataFrame({'Parameter Body': [test_suite['Parameter Body'][i] for i in true_index],
                               'Predictions': [list(problem.pred_results[i].numpy()) for i in true_index],
                               'Test Result': [test_suite['Test Result'][i] for i in true_index]
                               })
        ts_sub.to_csv(
            directory + 'Run_{}/{}/{}/Selection_{}/test_suites_{}.csv'.format(RUN,
                                                                              transfer_version,
                                                                              search_algorithm,
                                                                              ID, i),
            mode='w',
            header=True,
            index=False)

    # Save results to file
    print_function_values_to_file(fronts,
                                  directory + 'Run_{}/{}/{}/Selection_{}/FUN.'.format(
                                      RUN, transfer_version,
                                      search_algorithm,
                                      ID) + algorithm.label)
    print_variables_to_file(fronts,
                            directory + 'Run_{}/{}/{}/Selection_{}/VAR.'.format(RUN,
                                                                                transfer_version,
                                                                                search_algorithm,
                                                                                ID) + algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problems: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')


def select_test_suite(size=1000):
    test_suite = CANCER_DATASET.sample(n=size)
    test_suite.reset_index(drop=True, inplace=True)
    return test_suite


def load_test_selection_strategy(ts_size=100, RUN=1, ID=1000, transfer_version='v01_v03', model_path='',
                                 search_algorithm='IBEA',
                                 evaluation=12000):
    model = load_pretrained_model(model_path=model_path)
    test_suite = select_test_suite(size=ts_size)

    # gs_based_selection(model=model, test_suite=test_suite, ID=ID)
    search_based_selection(model=model, test_suite=test_suite, RUN=RUN, ID=ID,
                           transfer_version=transfer_version,
                           search_algorithm=search_algorithm,
                           evaluation=evaluation)


def test_suite_execution(test_suite):
    print('Test suites execution...')
    url = "http://localhost:8090/login"
    payload = 'username=guri-admin&password=guriadmin'

    s = requests.Session()
    s.auth = ("guri-admin", "guriadmin")
    s.post(url, data=payload)

    response_list = []
    for body in test_suite:
        try:
            a = s.post('http://localhost:8090/api/messages/validation', json=eval(body))
            response_list.append(a.json())
        except Exception as e:
            a = 'exception'
            response_list.append(a)
    return response_list


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
        parameter_body = eval(test_results['Parameter Body'][i])
        test_result = eval(test_results['Test Result'][i])

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
    parser.add_argument("--evaluation", type=int, default=30000)
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--search_algorithm", type=str, default='IBEA')
    args = parser.parse_args()
    #

    for t_v in [args.transfer]:
        print('============================================')
        print('Output directory: /global/D1/homes/qinghua/guri/test_suites_v2/'
              'Run_{}/{}/{}/Selection_{}'.format(args.run, t_v, args.search_algorithm, args.size))
        print('============================================')
        version_before = t_v.split('_')[0]
        version_after = t_v.split('_')[1]

        # print('Sub test suite size: {}'.format(args.size))
        start = time.time()
        load_test_selection_strategy(ts_size=args.size, RUN=args.run, ID=args.size, transfer_version=t_v,
                                     model_path='./save_model/guri_dt_model_{}.tf'.format(version_before),
                                     search_algorithm=args.search_algorithm,
                                     evaluation=args.evaluation)
        end = time.time()
        print('Test suite selection time is {} h.'.format((end - start) / 60 / 60))
