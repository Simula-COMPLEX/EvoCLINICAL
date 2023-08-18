#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 26/02/2023 15:37
# @Author  : Chengjie
# @File    : data_model.py
# @Software: PyCharm
import random
import time
import pandas as pd
import requests
from systemfiles.data_model_derived_from_rules import message_model_derived_from_rules
from systemfiles.result_code_map import r_map
import os

message_example = {'meldingID': 'V70Aq9NxYR', 'messageNumberId': '', 'messageVersion': 'Om_xBUk7ZaBYgU',
                   'topografi': '60',
                   'lokalisasjon': '1621', 'ekstralokalisasjon': '4s9Rqg7F8Mdls70', 'morfologi': '814023',
                   'basis': 'u', 'multiplisitet': 'eANuVern', 'metastase': '1', 'ds': '2',
                   'kirurgi': 'rBe3zrqAPXey_N', 'meldingsdato': '2041-10-22T08:30:57', 'meldingstype': '0g2Wq',
                   'stralebehandling': 'mGsdd', 'kjemoterapi': 'M1ayBZ3oHfupk', 'hormonbehandling': '3AmIwk1eRL3aILe2',
                   'kjonn': 'M', 'annenBehandling': 'PM5u2', 'diagnosedato': '',
                   'fodselsdato': '2028-10-29T03:29:22', 'mprio': 'q_3ffwssJmaHuUe', 'pt': 'QwUiVtjKVwoR2oJT',
                   'ct': 'FZU4QFVr_9ADq', 'ypt': 'FMIJlObblfpyiG', 'cn': 'oZ9oel1qi6HkqWk', 'pn': 'QnqSiv5k8JbXJVU',
                   'ypn': 'xgavWtn6', 'cm': 'cqip', 'pm': 'Klt4rg', 'ypm': '5nGAQixy', 'patologiStadium': 'W2Z59BDOB_l',
                   'kliniskStadium': 'JbY1hRIGQxwH7D0B', 'side': 'qriKQYn1G', 'meldingsnr': 'lxETrbk',
                   'sykehuskode': 'Lwc9VuMYh',
                   'topografiICDO3': 'uGOro4fLMnI6IcLy', 'morfologiICDO3': 'Ci82AcH7J',
                   'dodsdato': '2043-09-28T21:11:21',
                   'obduksjonstype': '3cpVm', 'fnr': 'hDvIP', 'tidMottatt': '2062-10-18T13:32:20',
                   'sorteringsKode': '7UTcO',
                   'meldingsstatus': 'qa6h', 'snomedM': 'AfLuMZLEByzr', 'snomedT': 'X'}

message_model = {
    'meldingID': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'messageNumberId': {
        'length': 20,
        'type': 'str',
        'value': ''
    },
    'messageVersion': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'topografi': {
        'length': 3,
        'type': 'str',
        'value': 'number'
    },
    'lokalisasjon': {
        'length': 4,
        'type': 'str',
        'value': 'number'
    },
    'ekstralokalisasjon': {
        'length': 4,
        'type': 'str',
        'value': 'number'
    },
    'morfologi': {
        'length': 6,
        'type': 'str',
        'value': 'number'
    },
    'basis': {
        'length': 2,
        'type': 'str',
        'value': 'number'
    },
    'multiplisitet': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'metastase': {
        'length': 1,
        'type': 'str',
        'value': 'numberORletter'
    },
    'ds': {
        'length': 1,
        'type': 'str',
        'value': 'number'
    },
    'kirurgi': {
        'length': 2,
        'type': 'str',
        'value': 'number'
    },
    'meldingsdato': {
        'length': 10,
        'type': 'date'
    },
    'meldingstype': {
        'length': 1,
        'type': 'str',
        'value': 'letter'
    },
    'stralebehandling': {
        'length': 1,
        'type': 'str',
        'value': 'number'
    },
    'kjemoterapi': {
        'length': 1,
        'type': 'str',
        'value': 'number'
    },
    'hormonbehandling': {
        'length': 1,
        'type': 'str',
        'value': 'number'
    },
    'kjonn': {
        'length': 1,
        'type': 'str',
        'value': 'letter'
    },
    'annenBehandling': {
        'length': 1,
        'type': 'str',
        'value': 'numberORletter'
    },
    'diagnosedato': {
        'length': 10,
        'type': 'date'
    },
    'fodselsdato': {
        'length': 10,
        'type': 'date'
    },
    'mprio': {
        'length': 8,
        'type': 'str',
        'value': ''
    },
    'pt': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'ct': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'ypt': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'cn': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'pn': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'ypn': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'cm': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'pm': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'ypm': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'patologiStadium': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'kliniskStadium': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'side': {
        'length': 5,
        'type': 'str',
        'value': ''
    },
    'meldingsnr': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'sykehuskode': {
        'length': 4,
        'type': 'str',
        'value': ''
    },
    'topografiICDO3': {
        'length': 3,
        'type': 'str',
        'value': ''
    },
    'morfologiICDO3': {
        'length': 6,
        'type': 'str',
        'value': ''
    },
    'dodsdato': {
        'length': 10,
        'type': 'date',
        'value': ''
    },
    'obduksjonstype': {
        'length': 2,
        'type': 'str',
        'value': ''
    },
    'fnr': {
        'length': 11,
        'type': 'str',
        'value': ''
    },
    'tidMottatt': {
        'length': 10,
        'type': 'date',
        'value': ''
    },
    'sorteringsKode': {
        'length': 4,
        'type': 'str',
        'value': ''
    },
    'meldingsstatus': {
        'length': 20,
        'type': 'str',
        'value': ''
    },
    'snomedM': {
        'length': 30,
        'type': 'str',
        'value': ''
    },
    'snomedT': {
        'length': 30,
        'type': 'str',
        'value': ''
    }
}

para_body = {'user': {'type': 'fixed', 'value': 'User'},
             'environment': {'type': 'enumerate', 'value': ['CRN', 'Test', 'Prod']},
             'activationDate': {'length': 10, 'type': 'date'},
             'cancerType': {'type': 'enumerate',
                            'value': ['Alle krefttyper', 'Colorectal', 'Lunge', 'Mamma', 'Prostata']},
             'cancerTypes': {'type': 'enumerate list',
                             'value': ['Alle krefttyper', 'Colorectal', 'Lunge', 'Mamma', 'Prostata']},
             'cancerMessages': message_model}

variables_for_training = {
    'topografi': {
        'length': 3,
        'type': 'str',
        'value': 'number'
    },
    'lokalisasjon': {
        'length': 4,
        'type': 'str',
        'value': 'number'
    },
    'ekstralokalisasjon': {
        'length': 4,
        'type': 'str',
        'value': 'number'
    },
    'morfologi': {
        'length': 6,
        'type': 'str',
        'value': 'number'
    },
    'basis': {
        'length': 2,
        'type': 'str',
        'value': 'number'
    },
    'multiplisitet': {
        'length': 10,
        'type': 'str'
    },
    'metastase': {
        'length': 1,
        'type': 'str',
        'value': 'numberORletter'
    },
    'ds': {
        'length': 1,
        'type': 'str',
        'value': 'number'
    },
    'kirurgi': {
        'length': 2,
        'type': 'str',
        'value': 'number'
    },
    'meldingsdato': {
        'length': 10,
        'type': 'date'
    },
    'meldingstype': {
        'length': 1,
        'type': 'str',
        'value': 'letter'
    },
    'stralebehandling': {
        'length': 1,
        'type': 'str',
        'value': 'number'
    },
    'kjemoterapi': {
        'length': 1,
        'type': 'str',
        'value': 'number'
    },
    'hormonbehandling': {
        'length': 1,
        'type': 'str',
        'value': 'number'
    },
    'kjonn': {
        'length': 1,
        'type': 'str',
        'value': 'letter'
    },
    'annenBehandling': {
        'length': 1,
        'type': 'str',
        'value': 'numberORletter'
    },
    'diagnosedato': {
        'length': 10,
        'type': 'date'
    },
    'fodselsdato': {
        'length': 10,
        'type': 'date'
    },
    'pt': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'ct': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'ypt': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'cn': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'pn': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'ypn': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'cm': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'pm': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'ypm': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'patologiStadium': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'kliniskStadium': {
        'length': 10,
        'type': 'str',
        'value': ''
    },
    'side': {
        'length': 5,
        'type': 'str',
        'value': ''
    },
    'sykehuskode': {
        'length': 4,
        'type': 'str',
        'value': ''
    },
    'topografiICDO3': {
        'length': 3,
        'type': 'str',
        'value': ''
    },
    'morfologiICDO3': {
        'length': 6,
        'type': 'str',
        'value': ''
    },
    'dodsdato': {
        'length': 10,
        'type': 'date',
        'value': ''
    }
}

# alphabet_list = []
# for i in range(0, 26):
#     l = chr(ord('a') + i)
#     alphabet_list.append(l)
# print(alphabet_list)

number_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

c_alphabet_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']

c_number_alphabet_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                          'J', 'K', 'L', 'M',
                          'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                          'U', 'V', 'W', 'X', 'Y', 'Z']

number_alphabet_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                        'J', 'K', 'L', 'M',
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                        'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def create_data_model():
    for key in message_model.keys():
        if key in variables_for_training.keys():
            message_model.update({key: variables_for_training[key]})
        else:
            message_model[key].update({'value': ''})


def generate_date(variable, control=False, end=(2011, 1, 1, 0, 0, 0, 0, 0, 0)):
    a1 = (1990, 1, 1, 0, 0, 0, 0, 0, 0)  # start time tuple:（1976-01-01 00：00：00）
    a2 = (2050, 12, 31, 23, 59, 59, 0, 0, 0)  # end time tuple:（2050-12-31 23：59：59)

    if variable == 'activationDate':
        a1 = (1990, 11, 24, 0, 0, 0, 0, 0, 0)  # start time tuple:（1976-01-01 00：00：00
    elif variable in ['meldingsdato', 'diagnosedato']:
        a1 = (1990, 1, 1, 0, 0, 0, 0, 0, 0)  # start time tuple:（1976-01-01 00：00：00

    if control:
        a2 = end

    start = time.mktime(a1)
    end = time.mktime(a2)
    t = random.randint(int(start), int(end))
    date_tuple = time.localtime(t)
    date = time.strftime('%Y-%m-%dT%H:%M:%S', date_tuple)
    return date


def generate_variable_value(variable, var_value_spec):
    var_value = ''
    '''
    generate variable value
    :param var_value_spec: 'type': ['fixed', 'str']
    :return: var_value: value of the variable
    '''

    if var_value_spec['type'] == 'fixed':
        var_value = var_value_spec['value']

    elif var_value_spec['type'] == 'enumerate':
        var_value = var_value_spec['value'][random.randint(0, len(var_value_spec['value']) - 1)]

    elif var_value_spec['type'] == 'enumerate list':
        # variable: cancerTypes
        list_size = random.randint(1, len(var_value_spec['value']) - 1)
        if list_size == len(var_value_spec['value']) or list_size == len(var_value_spec['value']) - 1:
            var_value = [var_value_spec['value'][0]]
        else:
            var_value = random.sample(var_value_spec['value'][1:], list_size)

    elif var_value_spec['type'] == 'date':
        var_value = generate_date(variable)

    return var_value


def generate_message(variable, m_model):
    cancer_message_dict = {}

    for key in m_model.keys():
        var_value = ''
        if m_model[key]['type'] == 'date':
            var_value = generate_date(variable)
        elif m_model[key]['type'] == 'str':
            if m_model[key]['value'] == '':
                var_value = ''.join(random.sample(number_alphabet_list, m_model[key]['length']))
                # for i in range(0, m_model['length']):
                #     var_value = var_value + str()
            elif m_model[key]['value'] == 'number':
                var_value = ''.join(random.sample(number_list, m_model[key]['length']))
            elif m_model[key]['value'] == 'letter':
                var_value = ''.join(random.sample(c_alphabet_list, m_model[key]['length']))
            elif m_model[key]['value'] == 'numberORletter':
                var_value = ''.join(random.sample(c_number_alphabet_list, m_model[key]['length']))
            elif m_model[key]['value'] == 'str':
                var_value = ''.join(random.sample(number_alphabet_list, m_model[key]['length']))

        cancer_message_dict.update({key: var_value})
    return [cancer_message_dict]


def generate_message_from_derived_model(variable):
    cancer_message_dict = {}
    for key in message_model_derived_from_rules.keys():
        var_value = ''
        if not message_model_derived_from_rules[key]['derived']:
            if message_model_derived_from_rules[key]['type'] == 'date':
                var_value = generate_date(variable)
            elif message_model_derived_from_rules[key]['type'] == 'str':
                if message_model_derived_from_rules[key]['value'] == '':
                    var_value = ''.join(
                        random.sample(number_alphabet_list, message_model_derived_from_rules[key]['length']))
                    # for i in range(0, m_model['length']):
                    #     var_value = var_value + str()
                elif message_model_derived_from_rules[key]['value'] == 'number':
                    var_value = ''.join(random.sample(number_list, message_model_derived_from_rules[key]['length']))
                elif message_model_derived_from_rules[key]['value'] == 'letter':
                    var_value = ''.join(random.sample(c_alphabet_list, message_model_derived_from_rules[key]['length']))
                elif message_model_derived_from_rules[key]['value'] == 'numberORletter':
                    var_value = ''.join(
                        random.sample(c_number_alphabet_list, message_model_derived_from_rules[key]['length']))
                elif message_model_derived_from_rules[key]['value'] == 'str':
                    var_value = ''.join(
                        random.sample(number_alphabet_list, message_model_derived_from_rules[key]['length']))
        else:
            possible_values = message_model_derived_from_rules[key]['possible_values']
            if possible_values == '':
                var_value = ''
            else:
                from random import choice
                candidate = list(choice(possible_values))
                for i in range(len(candidate)):
                    # print(candidate[i])
                    if candidate[i] == 'X':
                        candidate[i] = str(random.randint(0, 9))
                var_value = ''.join(candidate)
        cancer_message_dict.update({key: var_value})
    return [cancer_message_dict]


def generate_message_from_rule_mapping(rule_id):
    variable_ranges = r_map[rule_id]['variable_values_info']
    update_vars = {}
    for var in variable_ranges.keys():
        if variable_ranges[var]['derived'] is False:
            continue
        else:
            possible_values = variable_ranges[var]['possible_values']
            if possible_values == '':
                var_value = ''
            elif possible_values == 'NotNull':
                continue
            else:
                if possible_values[0] == 'Not':
                    # possible_values = possible_values[1:]
                    continue
                elif possible_values[0] == 'Less':
                    # date < []
                    var_value = generate_date('diagnosedato', control=True, end=(2011, 1, 1, 0, 0, 0, 0, 0, 0))
                else:
                    from random import choice
                    candidate = list(choice(possible_values))
                    for i in range(len(candidate)):
                        # print(candidate[i])
                        if candidate[i] == 'X':
                            if variable_ranges[var]['value'] == 'number':
                                candidate[i] = str(random.randint(0, 9))
                            elif variable_ranges[var]['value'] == '':
                                candidate[i] = str(random.sample(number_alphabet_list, 1)[0])
                    var_value = ''.join(candidate)
        update_vars.update({var: var_value})
    return update_vars


def generate_para_body():
    para_body_instance = {}
    for variable in para_body.keys():
        if variable == 'cancerMessages':
            flag = random.random()
            if flag < 0:
                var_value = generate_message(variable, para_body[variable])
            else:
                var_value = generate_message_from_derived_model(variable)

                # # generate variable values from result code map
                """
                for version v07
                """
                # rules = ['V04', 'V05', 'V09', 'V11', 'V12', 'V15', 'V16', 'V18', 'V29', 'V32',
                #          'V33', 'V34', 'V36', 'V37', 'V39', 'V41', 'V44', 'V45', 'V52', 'V53',
                #          'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63',
                #          'V66', 'V67', 'V68', 'V69', 'V70', 'V73']
                """
                for version v01
                """
                rules = ['V04', 'V05', 'V09', 'V11', 'V15', 'V16',
                         'V33', 'V39', 'V41', 'V44', 'V45', 'V47']
                # rules = ['V15', 'V29', 'V49']
                # rule_subset = random.sample(rules, random.randint(0, len(rules)))
                # if not rule_subset:
                #     continue
                # random.shuffle(rule_subset)

                rule_subset = ['V15']

                for rule in rule_subset:
                    replace_values = generate_message_from_rule_mapping(rule)
                    for key in replace_values.keys():
                        if key in var_value[0].keys():
                            var_value[0].update({key: replace_values[key]})

        else:
            var_value = generate_variable_value(variable, para_body[variable])

        para_body_instance.update({variable: var_value})

    return para_body_instance


def generate_guri_validation_data(version, file_id):
    # headers = {
    #     "Cookie": "JSESSIONID=A495892D82FE0A5634B52F495781A49E; Path=/; HttpOnly;"
    # }

    url = "http://localhost:8090/login"
    payload = 'username=guri-admin&password=guriadmin'

    s = requests.Session()
    s.auth = ("guri-admin", "guriadmin")
    response = s.post(url, data=payload)

    body_list = []
    output_list = []

    if not os.path.exists('./dataset_{}'.format(version)):
        os.makedirs('./dataset_{}'.format(version))

    file_n = './dataset_{}/test_results{}.csv'.format(version, file_id)
    print(file_n)

    # title = ['Parameter Body', 'Test Result']
    # pd.DataFrame([title]).to_csv(file_n, mode='w', index=False, header=False)

    for i in range(2000):
        if i % 100 == 0:
            print(i)
        body = generate_para_body()
        # print(json.dumps(body, indent=4))
        # print(body)
        body_list.append(body)
        try:
            a = s.post('http://localhost:8090/api/messages/validation', json=body)
            # print(a.status_code)
            output_list.append(a.json())
            pd.DataFrame([[body, a.json()]]).to_csv(file_n, mode='a', index=False, header=False)
        except Exception as e:
            # print(i)
            output_list.append('exception')
            pd.DataFrame([[body, 'exception']]).to_csv(file_n, mode='a', index=False, header=False)

    # save_dict.update({'Parameter Body': body_list, 'Test Result': output_list})
    # pd.DataFrame(save_dict).to_csv('./test_results3.csv', mode='w', index=False, header=True)


def create_training_dataset(version, file_id):
    file_name = './dataset_{}/training_dataset{}.csv'.format(version, file_id)
    title = ['environment', 'activationDate', 'cancerType', 'cancerTypes']
    title = title + list(variables_for_training.keys())

    validation_rules = pd.read_excel('./rules/xlsx/Validering_rule_{}.xlsx'.format(version),
                                     sheet_name='Validering rule')
    validation_rule_id = list(validation_rules['Rule Number'])
    validation_rule_id = [r_id for r_id in validation_rule_id if r_id == r_id]
    if 'V07' in validation_rule_id:
        validation_rule_id.remove('V07')

    title = title + list(validation_rule_id)

    print(title)
    pd.DataFrame([title]).to_csv(file_name, mode='w', index=False, header=False)

    test_results = pd.read_csv(filepath_or_buffer='./dataset_{}/test_results{}.csv'.format(version, file_id), sep=',')

    for i in range(len(test_results)):
        # parameter_body = json.loads(test_results['Parameter Body'][i].replace('\'', '\"'))
        if str(test_results['Test Result'][i]) == 'exception':
            continue
        parameter_body = eval(test_results['Parameter Body'][i])
        test_result = eval(test_results['Test Result'][i])

        if 'cancerMessages' not in parameter_body.keys():
            continue

        if test_result['preAggregationResult'] is None:
            model_input = [parameter_body['environment'], parameter_body['activationDate'],
                           parameter_body['cancerType'], parameter_body['cancerTypes']]
            for key in variables_for_training.keys():
                model_input.append(str(parameter_body['cancerMessages'][0][key]) if
                                   parameter_body['cancerMessages'][0][key] != '' else 'NULL or NONE')

            # print(test_result)
            for j in range(len(validation_rule_id)):
                # print(validation_rule_id[j])
                # print(len(test_result['results'][0]['messageValidationRuleResults']))
                for result in test_result['results'][0]['messageValidationRuleResults']:
                    if result['ruleNumber'] == validation_rule_id[j]:
                        # print(result['ruleNumber'], validation_rule_id[j])
                        model_input.append(result['resultCode'])
                        break

            pd.DataFrame([model_input]).to_csv(file_name, mode='a', index=False, header=False)


def reorder_headers(version, file_id):
    data = pd.read_csv(filepath_or_buffer='./dataset_{}/training_dataset{}.csv'.format(version, file_id), sep=',')
    reordered_headers = [
        'environment', 'cancerType', 'cancerTypes',
        'metastase', 'meldingstype', 'kjonn', 'annenBehandling',  # Categorical Variables
        'ds', 'stralebehandling',
        'kjemoterapi', 'hormonbehandling', 'kirurgi', 'topografi', 'lokalisasjon', 'ekstralokalisasjon', 'morfologi',
        'basis',  # numerical
        'activationDate', 'meldingsdato', 'diagnosedato', 'fodselsdato', 'dodsdato',  # date
        'multiplisitet', 'pt', 'ct', 'ypt', 'cn', 'pn', 'ypn', 'cm', 'pm', 'ypm',
        'patologiStadium', 'kliniskStadium', 'side', 'sykehuskode', 'topografiICDO3', 'morfologiICDO3',  # string
    ]
        # 'V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V08', 'V09', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
        # 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31',
        # 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47',
        # 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63',
        # 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V73']

    validation_rules = pd.read_excel('./rules/xlsx/Validering_rule_{}.xlsx'.format(version),
                                     sheet_name='Validering rule')
    validation_rule_id = list(validation_rules['Rule Number'])
    validation_rule_id = [r_id for r_id in validation_rule_id if r_id == r_id]
    if 'V07' in validation_rule_id:
        validation_rule_id.remove('V07')

    print(validation_rule_id)

    reordered_headers = reordered_headers + validation_rule_id

    print(len(reordered_headers))
    data = data[reordered_headers]

    print(len(data))

    data.to_csv('./dataset_{}/training_dataset_reordered{}.csv'.format(version, file_id), mode='w', index=True, header=True)


def check_rule(version, file_id):
    f_n = './dataset_{}/training_dataset{}.csv'.format(version, file_id)
    dataset = pd.read_csv(filepath_or_buffer=f_n, sep=',')
    rule_headers = ['V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V08', 'V09', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
                    'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
                    'V30', 'V31',
                    'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45',
                    'V46', 'V47',
                    'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61',
                    'V62', 'V63',
                    'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V73']

    validation_rules = pd.read_excel('./rules/xlsx/Validering_rule_{}.xlsx'.format(version),
                                     sheet_name='Validering rule')
    validation_rule_id = list(validation_rules['Rule Number'])
    validation_rule_id = [r_id for r_id in validation_rule_id if r_id == r_id]
    if 'V07' in validation_rule_id:
        validation_rule_id.remove('V07')

    rule_headers = validation_rule_id

    check_results = {}
    check_results.update({'ruleID': ['resultCode[1]: INFO', 'resultCode[2]: NOT_APPLIED',
                                     'resultCode[3]: ERROR', 'resultCode[4]: WARNING',
                                     '#Message(FailedRules)', '#Message(PassRules)']})
    fail_all_headers = []
    for rule_header in rule_headers:
        check_results.update({rule_header: [list(dataset[rule_header]).count(1), list(dataset[rule_header]).count(2),
                                            list(dataset[rule_header]).count(3), list(dataset[rule_header]).count(4),
                                            list(dataset[rule_header]).count(2) + list(dataset[rule_header]).count(3),
                                            list(dataset[rule_header]).count(1) + list(dataset[rule_header]).count(4)]})

    #     fail_all_headers.append(rule_header) if list(dataset[rule_header]).count(1) + list(dataset[rule_header]).count(
    #         4) != 0 else 0
    #
    # print(fail_all_headers, len(fail_all_headers))

    # ['V02', 'V03', 'V04', 'V05', 'V09', 'V10', 'V12', 'V15', 'V16', 'V18', 'V29', 'V31', 'V32', 'V33', 'V34', 'V36',
    #  'V37', 'V38', 'V39', 'V41', 'V44', 'V45', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61',
    #  'V62', 'V63', 'V64', 'V65', 'V66', 'V68', 'V69', 'V70', 'V73']
    pd.DataFrame(check_results).to_csv('./dataset_{}/check_rule_results_dataset{}.csv'.format(version, file_id),
                                       mode='w', header=True, index=False)


def merge_dataset(*args):
    rule_headers = ['V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V08', 'V09', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
                    'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29',
                    'V30', 'V31',
                    'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45',
                    'V46', 'V47',
                    'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61',
                    'V62', 'V63',
                    'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V73']

    validation_rules = pd.read_excel('./rules/xlsx/Validering_rule_{}.xlsx'.format(args[len(args)-1]),
                                     sheet_name='Validering rule')
    validation_rule_id = list(validation_rules['Rule Number'])
    rule_headers = [r_id for r_id in validation_rule_id if r_id == r_id]
    if 'V07' in rule_headers:
        rule_headers.remove('V07')

    target = args[0]
    target_df = pd.read_csv(filepath_or_buffer=target, sep=',', index_col=0)

    print(len(target_df))

    for f_n in args[1:-1]:
        print(f_n)
        f_n_df = pd.read_csv(filepath_or_buffer=f_n, sep=',', index_col=0)
        for idx, row in f_n_df.iterrows():
            if idx % 500 == 0:
                print(idx)
            add = False
            for rule in rule_headers:
                if row[rule] in [1, 3, 4]:
                    add = True
            if add is True:
                # target_df.conc(row)
                # target_df = pd.concat([row, target_df.loc[:]]).reset_index(drop=True)
                target_df.loc[len(target_df.index)] = row
                # break
    print(target_df, len(target_df))

    target_df.to_csv('./dataset_{}/training_dataset_reordered102.csv'.format(args[len(args)-1]),
                     mode='w', header=True, index=True)


def convert_variable_types(f_n='./dataset_v01/training_dataset_reordered1.csv'):
    dataset = pd.read_csv(filepath_or_buffer=f_n, sep=',', index_col=0)
    column = {'kirurgi': 2, 'topografi': 3, 'lokalisasjon': 4, 'ekstralokalisasjon': 4, 'morfologi': 6, 'basis': 2}
    for i in range(len(dataset)):
        for var in column.keys():
            variable_values = list(str(dataset[var][i]))
            if len(variable_values) < column[var]:

                dataset.loc[i, var] = str(''.join(['0' for n in range(column[var] - len(variable_values))]) +
                                          ''.join(variable_values))
            else:
                dataset.loc[i, var] = str(dataset[var][i])
    dataset.to_csv(f_n, mode='w', index=True, header=True)


if __name__ == '__main__':
    file_i = '_final'
    ver = 'v70'
    # generate_guri_validation_data(ver, file_i)
    create_training_dataset(ver, file_i)
    # reorder_headers(ver, file_i)
    check_rule(ver, file_i)
