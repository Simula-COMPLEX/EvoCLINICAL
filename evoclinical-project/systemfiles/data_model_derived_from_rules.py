#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22/03/2023 23:15
# @Author  : Chengjie
# @File    : data_model_derived_from_rules.py
# @Software: PyCharm
import json

message_model_derived_from_rules = {
    "meldingID": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "messageNumberId": {
        "length": 20,
        "type": "str",
        "value": "",
        "derived": False
    },
    "messageVersion": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "topografi": {
        "length": 3,
        "type": "str",
        "value": "number",
        "derived": True,
        "possible_values": [
            "420",
            "421",
            "76X",
            "752",
            "809",
            "220",
            "50X",
            "60X",
            "61X",
            "62X",
            "63X",
            "67X",
            "73X",
            "25X",
            "34X",
            "48X",
            "51X",
            "52X",
            "53X",
            "54X",
            "55X",
            "56X",
            "57X"
        ]
    },
    "lokalisasjon": {
        "length": 4,
        "type": "str",
        "value": "number",
        "derived": True,
        "possible_values": [
            "1708",
            "1573",
            "1559",
            "1770",
            "1801",
            "1802",
            "1808",
            "1810",
            "1811",
            "1813",
            "1814",
            "1818",
            "1819",
            "1994",
            "1930",
            "1931",
            "1932",
            "1933",
            "1934",
            "1937",
            "1938",
            "1939",
            "1953",
            "1954",
            "1955",
            "1705",
            "1535",
            "1545",
            "1550",
            "181X",
            "194X",
            "206X",
            "207X",
            "190X"
        ]
    },
    "ekstralokalisasjon": {
        "length": 4,
        "type": "str",
        "value": "number",
        "derived": True,
        "possible_values": [
            "8888", "XXXX"
        ]
    },
    "morfologi": {
        "length": 6,
        "type": "str",
        "value": "number",
        "derived": True,
        "possible_values": [
            "935019",
            "9990XX",
            "815XXX",
            "824XXX",
            "868XXX",
            "869XXX",
            "8700XXX",
            "XXXX3X",
            "XXXX0X",
            "XXXX1X",
            "XXXX2X",
            "81903X",
            "85503X",
            "872XXX",
            "873XXX",
            "874XXX",
            "875XXX",
            "876XXX",
            "877XXX",
            "878XXX",
            "879XXX",
            "8970XX"
        ]
    },
    "basis": {
        "length": 2,
        "type": "str",
        "value": "number",
        "derived": True,
        "possible_values": [
            "29",
            "33",
            "38",
            "45",
            "47",
            "36",
            "60",
            "98",
            "57",
            "79",
            "00",
            "10",
            "20",
            "23",
            "29",
            "30",
            "31",
            "40",
            "90",
            "22",
            "32",
            "36",
            "38",
            "60",
            "70",
            "72",
            "74",
            "75",
            "76"
        ]
    },
    "multiplisitet": {
        "length": 10,
        "type": "str",
        "derived": True,
        "possible_values": ""
    },
    "metastase": {
        "length": 1,
        "type": "str",
        "value": "numberORletter",
        "derived": True,
        "possible_values": [
            "0",
            "A",
            "B",
            "C",
            "D",
            "9"
        ]
    },
    "ds": {
        "length": 1,
        "type": "str",
        "value": "number",
        "derived": False
    },
    "kirurgi": {
        "length": 2,
        "type": "str",
        "value": "number",
        "derived": True,
        "possible_values": [
            "14",
            "11",
            "10",
            "01",
            "02",
            "96",
            "97"
        ]
    },
    "meldingsdato": {
        "length": 10,
        "type": "date",
        "derived": False
    },
    "meldingstype": {
        "length": 1,
        "type": "str",
        "value": "letter",
        "derived": True,
        "possible_values": [
            "O",
            "D",
            "S",
            "H",
            "K"
        ]
    },
    "stralebehandling": {
        "length": 1,
        "type": "str",
        "value": "number",
        "derived": False
    },
    "kjemoterapi": {
        "length": 1,
        "type": "str",
        "value": "number",
        "derived": False
    },
    "hormonbehandling": {
        "length": 1,
        "type": "str",
        "value": "number",
        "derived": False
    },
    "kjonn": {
        "length": 1,
        "type": "str",
        "value": "letter",
        "derived": True,
        "possible_values": [
            "M",
            "D",
            "S",
            "H",
            "K"
        ]
    },
    "annenBehandling": {
        "length": 1,
        "type": "str",
        "value": "numberORletter",
        "derived": False
    },
    "diagnosedato": {
        "length": 10,
        "type": "date",
        "derived": False
    },
    "fodselsdato": {
        "length": 10,
        "type": "date",
        "derived": False
    },
    "mprio": {
        "length": 8,
        "type": "str",
        "value": "",
        "derived": False
    },
    "pt": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "ct": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "ypt": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "cn": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "pn": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "ypn": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "cm": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "pm": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "ypm": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "patologiStadium": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "kliniskStadium": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "side": {
        "length": 5,
        "type": "str",
        "value": "",
        "derived": True,
        "possible_values": ""
    },
    "meldingsnr": {
        "length": 10,
        "type": "str",
        "value": "",
        "derived": False
    },
    "sykehuskode": {
        "length": 4,
        "type": "str",
        "value": "",
        "derived": False
    },
    "topografiICDO3": {
        "length": 3,
        "type": "str",
        "value": "",
        "derived": False
    },
    "morfologiICDO3": {
        "length": 6,
        "type": "str",
        "value": "",
        "derived": False
    },
    "dodsdato": {
        "length": 10,
        "type": "date",
        "value": "",
        "derived": False
    },
    "obduksjonstype": {
        "length": 2,
        "type": "str",
        "value": "",
        "derived": False
    },
    "fnr": {
        "length": 11,
        "type": "str",
        "value": "",
        "derived": False
    },
    "tidMottatt": {
        "length": 10,
        "type": "date",
        "value": "",
        "derived": False
    },
    "sorteringsKode": {
        "length": 4,
        "type": "str",
        "value": "",
        "derived": False
    },
    "meldingsstatus": {
        "length": 20,
        "type": "str",
        "value": "",
        "derived": False
    },
    "snomedM": {
        "length": 30,
        "type": "str",
        "value": "",
        "derived": False
    },
    "snomedT": {
        "length": 30,
        "type": "str",
        "value": "",
        "derived": False
    }
}

variables_for_training = {
    'topografi': {
        'length': 3,
        'type': 'str',
        'value': 'number',
        'derived': True,
        'possible_values': ['420', '421', '76X', '752', '809', '220',
                            '50X', '60X', '61X', '62X', '63X', '67X', '73X', '25X',
                            '34X', '48X', '51X', '52X', '53X', '54X', '55X', '56X', '57X']
    },
    'lokalisasjon': {
        'length': 4,
        'type': 'str',
        'value': 'number',
        'derived': True,
        'possible_values': ['1708', '1573', '1559', '1770', '1801', '1802', '1808', '1810', '1811', '1813',
                            '1814', '1818', '1819', '1994', '1930', '1931', '1932', '1933', '1934', '1937',
                            '1938', '1939', '1953', '1954', '1955', '1705', '1535', '1545', '1550',
                            '181X', '194X', '206X', '207X', '190X', ]
    },
    'ekstralokalisasjon': {
        'length': 4,
        'type': 'str',
        'value': 'number',
        'derived': True,
        'possible_values': ['8888', 'XXXX']

    },
    'morfologi': {
        'length': 6,
        'type': 'str',
        'value': 'number',
        'derived': True,
        'possible_values': ['935019',
                            '9990XX', '815XXX', '824XXX', '868XXX', '869XXX', '870XXX', 'XXXX3X',
                            'XXXX0X', 'XXXX1X', 'XXXX2X', '81903X', '85503X',
                            '872XXX', '873XXX', '874XXX', '875XXX', '876XXX', '877XXX', '878XXX',
                            '879XXX', '8970XX']
    },
    'basis': {
        'length': 2,
        'type': 'str',
        'value': 'number',
        'derived': True,
        'possible_values': ['29', '33', '38', '45', '47', '36', '60', '98', '57', '79',
                            '00', '10', '20', '23', '29', '30', '31', '40', '90',
                            '22', '32', '36', '38', '60', '70', '72', '74', '75', '76',
                            ]
    },
    'multiplisitet': {
        'length': 10,
        'type': 'str',
        'derived': False,
        'possible_values': ''
    },
    'metastase': {
        'length': 1,
        'type': 'str',
        'value': 'numberORletter',
        'derived': True,
        'possible_values': ['0', 'A', 'B', 'C', 'D', '9']
    },
    'ds': {
        'length': 1,
        'type': 'str',
        'value': 'number',
        'derived': False
    },
    'kirurgi': {
        'length': 2,
        'type': 'str',
        'value': 'number',
        'derived': True,
        'possible_values': ['14', '11', '10', '01', '02', '96', '97']
    },
    'meldingsdato': {
        'length': 10,
        'type': 'date',
        'derived': False
    },
    'meldingstype': {
        'length': 1,
        'type': 'str',
        'value': 'letter',
        'derived': True,
        'possible_values': ['O', 'D', 'S', 'H', 'K']
    },
    'stralebehandling': {
        'length': 1,
        'type': 'str',
        'value': 'number',
        'derived': False
    },
    'kjemoterapi': {
        'length': 1,
        'type': 'str',
        'value': 'number',
        'derived': False
    },
    'hormonbehandling': {
        'length': 1,
        'type': 'str',
        'value': 'number',
        'derived': False
    },
    'kjonn': {
        'length': 1,
        'type': 'str',
        'value': 'letter',
        'derived': True,
        'possible_values': ['M', 'D', 'S', 'H', 'K']
    },
    'annenBehandling': {
        'length': 1,
        'type': 'str',
        'value': 'numberORletter',
        'derived': False
    },
    'diagnosedato': {
        'length': 10,
        'type': 'date',
        'derived': False
    },
    'fodselsdato': {
        'length': 10,
        'type': 'date',
        'derived': False
    },
    'pt': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'ct': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'ypt': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'cn': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'pn': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'ypn': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'cm': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'pm': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'ypm': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'patologiStadium': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'kliniskStadium': {
        'length': 10,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'side': {
        'length': 5,
        'type': 'str',
        'value': '',
        'derived': True,
        'possible_values': ''
    },
    'sykehuskode': {
        'length': 4,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'topografiICDO3': {
        'length': 3,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'morfologiICDO3': {
        'length': 6,
        'type': 'str',
        'value': '',
        'derived': False
    },
    'dodsdato': {
        'length': 10,
        'type': 'date',
        'value': '',
        'derived': False
    }
}