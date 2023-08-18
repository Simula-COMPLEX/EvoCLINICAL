#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 14/04/2023 17:08
# @Author  : Chengjie
# @File    : problem.py
# @Software: PyCharm
import difflib
import math
import random
from math import log2
import numpy as np
from jmetal.core.problem import IntegerProblem, BinaryProblem
from jmetal.core.solution import IntegerSolution, BinarySolution
import tensorflow as tf
from utils.preprocess import preprocess_para_body, get_para_body_df, rule_profile, \
    calculate_failed_coverage, calculate_output_diversity


def sigmoid_norm(x):
    return 1 / (1 + math.exp(-x))


def get_rule_results(test_results, version='v01'):
    validation_rule_id = rule_profile[version]

    result_codes = []

    for j in range(len(validation_rule_id)):
        for result in test_results['results'][0]['messageValidationRuleResults']:
            if result['ruleNumber'] == validation_rule_id[j]:
                result_codes.append(result['resultCode'] - 1)
                break

    return result_codes


def calculate_information_entropy(pred_softmax):
    pred_softmax = pred_softmax.numpy()

    def information_entropy(events, ets=1e-15):
        return -sum([p * log2(p + ets) for p in events])

    information_entropies = []
    for i in range(len(pred_softmax)):
        information_entropy_i = []
        for j in range(len(pred_softmax[i])):
            information_entropy_i.append(information_entropy(pred_softmax[i][j]))
        information_entropies.append(information_entropy_i)
    return tf.transpose(information_entropies)


def calculate_diversity(pred_results):
    pred_results = pred_results.numpy()
    s = set()
    for r in pred_results:
        s.add(tuple(r))
    unique = len(list(s))
    return unique / len(pred_results)


def calculate_diversity_for_all(all_features):
    def calculate_variable_div(f1, f2):
        var_div = 0
        for k in range(len(f1)):
            var_div += 1 if f1[k] != f2[k] else 0
        return var_div / len(f1)

    diversity_array = []
    diversity_array_2 = []
    for i in range(len(all_features) - 1):
        # diversity_array_i = []
        for j in range(i + 1, len(all_features)):
            div_i_j = calculate_variable_div(all_features[i], all_features[j])
            # diversity_array_i.append(div_i_j)
            diversity_array_2.append(div_i_j)
        # diversity_array.append(diversity_array_i)

    return diversity_array, diversity_array_2


def calculate_input_diversity_for_solution(diversity_all, index):
    # print(len(diversity_all))
    # print(index)
    diversity_solution = diversity_all[len(diversity_all) - 1]
    for i in range(len(index) - 1):
        slicing = index[i + 1:]
        diversity_solution += [diversity_all[index[i]][j - index[i] - 1] for j in slicing]

    return np.array(diversity_solution).mean()


def calculate_input_diversity_for_solution2(length, diversity_all, index):
    # print(len(diversity_all))
    # print(index)

    diversity_solution = 0
    for i in range(len(index) - 1):
        for j in range(i + 1, len(index)):
            ind = index[i] * length - index[i] * (index[i] - 1) / 2 + j - i - 1
            # print(ind)
            diversity_solution += diversity_all[int(ind)]

    return diversity_solution / ((len(index) - 1) * (len(index) - 2) / 2)


def calculate_uncertainty(info_entropy, index):
    uncertainties = []
    for ind in index:
        # uncertainty = np.array(info_entropy[ind]).mean()
        uncertainty = np.array(info_entropy[ind]).mean()
        uncertainties.append(uncertainty)
    return np.array(uncertainties).mean()


class TestSuiteSelection(BinaryProblem):
    def __init__(self, test_suite, guri_dt, result_code_weights, probability_distribution, search_algorithm='NSGA-II', number_of_bits: int = 100,
                 run=1, ID=1, transfer_version='v01_v03'):
        super(TestSuiteSelection, self).__init__()

        ts = test_suite['Parameter Body'].values
        ts_results = []
        for i in range(len(ts)):
            ts[i] = eval(ts[i])
            ts_results.append(get_rule_results(test_results=eval(test_suite['Test Result'][i]), version=transfer_version.split('_')[0]))

        self.SEARCH_ALGORITHM = search_algorithm
        self.number_of_bits = number_of_bits
        self.number_of_objectives = 5  # min (test suite size), max (diversity, coverage, uncertainty)
        self.number_of_variables = 1
        self.number_of_constraints = 1
        self.TS = ts
        self.TS_RESULTS = ts_results
        self.DT = guri_dt
        self.iteration = 0
        self.RUN = run
        self.ID = ID
        self.transfer_version = transfer_version
        self.R_C_WEIGHT = result_code_weights
        self.PROB_DIST = probability_distribution

        # predict message validation results
        self.INPUTS = [preprocess_para_body(self.TS)]

        _, self.INPUTS_DIVERSITY = calculate_diversity_for_all(get_para_body_df(self.TS).values)

        self.pred_softmax = tf.squeeze(guri_dt.predict(self.INPUTS))
        self.pred_results = tf.transpose(tf.squeeze(tf.argmax(self.pred_softmax, axis=-1)))

        # cov_arr, div_arr, coverage, div = calculate_weighted_coverage_o_diversity(self.pred_results, self.R_C_WEIGHT,
        #                                                                           self.PROB_DIST)

        self.info_entropy = calculate_information_entropy(self.pred_softmax)

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]

        # self.directory = '/Users/chengjielu/Work/FSE2023/TestGURI/rulevalidationpredict/multi_outputs_model/test_suites_v2/'
        #
        self.directory = '/global/D1/homes/qinghua/guri/test_suites_v2/'

        open(self.directory + '/Run_{}/{}/{}/Selection_{}/Middle_Fun_Result_Multi_'.format(
            self.RUN,
            self.transfer_version,
            self.SEARCH_ALGORITHM,
            self.ID)
             + str(self.number_of_variables) + '.txt', 'w', encoding='utf-8')

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        self.iteration += 1
        if self.iteration % 1000 == 0:
            print(self.iteration)
        # print(solution.variables[0])
        true_index = [i for i in range(len(solution.variables[0])) if solution.variables[0][i] is True]
        test_suite_size = len(true_index)
        if test_suite_size == 0:
            false_pred_coverage = 0
            i_diversity = 0
            o_diversity = 0
            uncertainty = 1
        else:
            # _, _, coverage, o_diversity = calculate_weighted_coverage_o_diversity(
            #     tf.gather(self.pred_results, indices=true_index), self.R_C_WEIGHT, self.PROB_DIST)
            _, o_diversity = calculate_output_diversity(tf.gather(self.pred_results, indices=true_index), self.PROB_DIST)
            false_pred_coverage = calculate_failed_coverage(tf.gather(self.pred_results, indices=true_index),
                                                            tf.gather(self.TS_RESULTS, indices=true_index))

            try:
                i_diversity = calculate_input_diversity_for_solution2(self.number_of_bits - 1, self.INPUTS_DIVERSITY,
                                                                      true_index)
            except ZeroDivisionError:
                i_diversity = 0

            uncertainty = calculate_uncertainty(self.info_entropy, true_index)

        solution.objectives[0] = test_suite_size / self.number_of_bits
        solution.objectives[1] = -false_pred_coverage  # -(coverage - 0.25) / (1 - 0.25)
        solution.objectives[2] = -(i_diversity - 0) / (1 - 0)
        solution.objectives[3] = (o_diversity - 0) / (1 - 0)
        solution.objectives[4] = -(uncertainty - 0) / (1 - 0)

        f = open(
            self.directory + '/Run_{}/{}/{}/Selection_{}/Middle_Fun_Result_Multi_'.format(
                self.RUN,
                self.transfer_version,
                self.SEARCH_ALGORITHM,
                self.ID)
            + str(self.number_of_variables) + '.txt', 'a', encoding='utf-8')
        f.writelines(
            str([i for i in solution.objectives]).replace('[', '').replace(']', '').replace("'", '').replace(
                ',',
                '') + '\n')
        f.close()

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)
        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]
        return new_solution

    def __evaluate_constraints(self, solution: BinarySolution) -> None:
        self.number_of_constraints = 1
        w1 = len([i for i in range(len(solution.variables[0])) if solution.variables[0][i] is True])

        solution.constraints[0] = w1 - 1

    def get_name(self) -> str:
        return 'TestSuiteSelection'
