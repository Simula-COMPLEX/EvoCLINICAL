r_map = {
    'V66': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': False
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '206X',
                    '207X'
                ]
            },
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': False
            },
            'diagnosedato': {
                'length': 10,
                'type': 'date',
                'derived': False
            },
            'pm': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V47': {
        'variable_values_info': {
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': 'NotNull'
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': 'NotNull'
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': 'NotNull'
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'derived': True,
                'possible_values': 'NotNull'
            },
            'kirurgi': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': 'NotNull'
            },
            'meldingsdato': {
                'length': 10,
                'derived': True,
                'possible_values': 'NotNull'
            },
            'sykehuskode': {
                'length': 4,
                'type': 'str',
                'derived': True,
                'possible_values': 'NotNull'
            }
        }
    },
    'V02': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '50X'
                ]
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    'A',
                    'B',
                    'C',
                    'D'
                ]
            }
        }
    },
    'V04': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '619'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '29'
                ]
            }
        }
    },
    'V05': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': False
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'Not',
                    '190X'
                ]
            },
            'ekstralokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'Not',
                    '8888'
                ]
            }
        }
    },
    'V73': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': False
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '190X'
                ]
            },
            'ekstralokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '8888'
                ]
            }
        }
    },
    'V53': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1535',
                    '1545'
                ]
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1'
                ]
            },
            'diagnosedato': {
                'length': 10,
                'type': 'date',
                'derived': True,
                'possible_values': [
                    'Less',
                    '01-01-2011'
                ]
            }
        }
    },
    'V09': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '50X',
                    '619',
                    '67X',
                    '739'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '181X',
                    '194X'
                ]
            },
            'kirurgi': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '14'
                ]
            }
        }
    },
    'V19': {
        'variable_values_info': {
            'stralebehandling': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '',
                    '0',
                    '1',
                    '2',
                    '3',
                    '4',
                    '5',
                    '9'
                ]
            }
        }
    },
    'V20': {
        'variable_values_info': {
            'hormonbehandling': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '',
                    '0',
                    '1',
                    '2',
                    '9'
                ]
            }
        }
    },
    'V64': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '76X',
                    '48X'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '206X',
                    '207X'
                ]
            },
            'multiplisitet': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': True,
                'possible_values': ['', 'XXXXXXXXXX']
            }
        }
    },
    'V22': {
        'variable_values_info': {
            'annenBehandling': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    '',
                    '0',
                    '1',
                    '2',
                    '3',
                    '4',
                    '9',
                    'J'
                ]
            }
        }
    },
    'V11': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': False
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'Not',
                    '206X',
                    '207X',
                    '190X'
                ]
            },
            'ekstralokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'Not',
                    '8888'
                ]
            },
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': False
            },
            'diagnosedato': {
                'length': 10,
                'type': 'date',
                'derived': False
            },
            'pm': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V13': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '50X',
                    '61X',
                    '73X'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '206X',
                    '207X'
                ]
            },
            'kirurgi': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '14',
                    '11',
                    '10',
                    '01',
                    '02',
                    '96',
                    '97'
                ]
            }
        }
    },
    'V15': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    # '763',
                    # '34X',
                    # '48X',
                    # '51X',
                    # '52X',
                    # '53X',
                    # '54X',
                    # '55X',
                    # '56X',
                    # '57X',
                    'XXX'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    # '190X',
                    'XXXX'
                ]
            },
            'ekstralokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    # '8888',
                    'XXXX'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '36',
                    '60'
                ]
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    '6'
                ]
            }
        }
    },
    'V51': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1708',
                    '1573',
                    '1559',
                    '1770',
                    '1801',
                    '1802',
                    '1808',
                    '1810',
                    '1811',
                    '1813',
                    '1814',
                    '1818',
                    '1819',
                    '1994',
                    '1930',
                    '1931',
                    '1932',
                    '1933',
                    '1934',
                    '1937',
                    '1938',
                    '1939',
                    '1953',
                    '1954',
                    '1955',
                    '1705',
                    '1535',
                    '1545',
                    '1550',
                    '181X',
                    '194X',
                    '206X',
                    '207X',
                    '190X'
                ]
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1',
                    '3'
                ]
            },
            'meldingstype': {
                'length': 1,
                'type': 'str',
                'value': 'letter',
                'derived': True,
                'possible_values': [
                    'S'
                ]
            }
        }
    },
    'V52': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '763',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '872XXX',
                    '873XXX',
                    '874XXX',
                    '875XXX',
                    '876XXX',
                    '877XXX',
                    '878XXX',
                    '879XXX'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '36',
                    '60'
                ]
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    '5'
                ]
            }
        }
    },
    'V14': {
        'variable_values_info': {
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '36',
                    '60'
                ]
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    'Not',
                    '0',
                    'D',
                    '9'
                ]
            }
        }
    },
    'V17': {
        'variable_values_info': {
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'XXXX3X'
                ]
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '3'
                ]
            }
        }
    },
    'V16': {
        'variable_values_info': {
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '9990XX'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '98'
                ]
            }
        }
    },
    'V67': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'Not',
                    '206X',
                    '207X'
                ]
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '5'
                ]
            }
        }
    },
    'V21': {
        'variable_values_info': {
            'kjemoterapi': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': False
            }
        }
    },
    'V24': {
        'variable_values_info': {
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'XXXX0X',
                    'XXXX1X',
                    'XXXX2X'
                ]
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '5'
                ]
            }
        }
    },
    'V23': {
        'variable_values_info': {
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'XXXX3X'
                ]
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '2'
                ]
            }
        }
    },
    'V25': {
        'variable_values_info': {
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': ['00', '10', '20', '23', '29', '30', '31', '40', '90']
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': ['5']
            }
        }
    },
    'V29': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'Not',
                    '206X',
                    '207X'
                ]
            },
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '9990XX'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '98'
                ]
            },
            'meldingstype': {
                'length': 1,
                'type': 'str',
                'value': 'letter',
                'derived': True,
                'possible_values': [
                    'O',
                    'H'
                ]
            }
        }
    },
    'V28': {
        'variable_values_info': {
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': ['22', '32', '36', '38', '60', '70', '72', '74', '75', '76']
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': ['1', '2', '3', '4', '7']
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': ['1']
            }
        }
    },
    'V40': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': ['Not', '481', '482', '488', '570', '569',
                                    '579', '619', '51X', '52X', '53X', '54X', '55X']
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    'Not',
                    '5'
                ]
            },
            'meldingstype': {
                'length': 1,
                'type': 'str',
                'value': 'letter',
                'derived': True,
                'possible_values': [
                    'K'
                ]
            }
        }
    },
    'V31': {
        'variable_values_info': {
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '98'
                ]
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    '0',
                    '9'
                ]
            }
        }
    },
    'V26': {
        'variable_values_info': {
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'XXXX0X',
                    'XXXX1X',
                    'XXXX2X'
                ]
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1'
                ]
            }
        }
    },
    'V33': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '752'
                ]
            },
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '935019'
                ]
            }
        }
    },
    'V01': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'Not',
                    '206X',
                    '207X'
                ]
            },
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': False
            },
            'diagnosedato': {
                'length': 10,
                'type': 'date',
                'derived': False
            }
        }
    },
    'V38': {
        'variable_values_info': {
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '9990XX'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '29',
                    '33',
                    '38',
                    '45',
                    '47',
                    '36',
                    '60',
                    '98',
                    '57',
                    '79',
                    '00',
                    '10',
                    '20',
                    '23',
                    '29',
                    '30',
                    '31',
                    '40',
                    '90',
                    # '22',
                    '32',
                    '36',
                    '38',
                    '60',
                    '70',
                    # '72',
                    '74',
                    '75',
                    '76'
                ]
            }
        }
    },
    'V34': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '809'
                ]
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    '4', '7', '9'
                ]
            }
        }
    },
    'V42': {
        'variable_values_info': {
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    '0', '1', '2', '3', '4', '5', '6', '7', '9', 'A', 'B', 'C', 'D'
                ]
            }
        }
    },
    'V35': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '206X',
                    '207X',
                    'XXX'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '29',
                    '33',
                    '38',
                    '45',
                    '47',
                    '36',
                    '60',
                    '98',
                    '57',
                    '79',
                    '00',
                    '10',
                    '20',
                    '23',
                    '29',
                    '30',
                    '31',
                    '40',
                    '90',
                    '22',
                    '32',
                    '36',
                    '38',
                    '60',
                    '70',
                    '72',
                    '74',
                    '75',
                    '76'
                ]
            },
            'kirurgi': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '14',
                    '11',
                    '10',
                    '01',
                    '02',
                    '96',
                    '97'
                ]
            },
            'meldingstype': {
                'length': 1,
                'type': 'str',
                'value': 'letter',
                'derived': True,
                'possible_values': [
                    'O', 'D', 'S', 'H', 'K'
                ]
            }
        }
    },
    'V43': {
        'variable_values_info': {
            'kirurgi': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'Not',
                    '95'
                ]
            }
        }
    },
    'V44': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '2070'
                ]
            },
            'annenBehandling': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    '2'
                ]
            }
        }
    },
    'V45': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1770'
                ]
            },
            'hormonbehandling': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '2'
                ]
            }
        }
    },
    'V49': {
        'variable_values_info': {
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'Not',
                    '98'
                ]
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    'Not',
                    '0'
                ]
            },
            'ds': {
                'length': 1,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '2',
                    '3'
                ]
            },
            'kirurgi': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '01',
                    '02',
                    '96',
                ]
            },
            'meldingstype': {
                'length': 1,
                'type': 'str',
                'value': 'letter',
                'derived': True,
                'possible_values': [
                    'H'
                ]
            }
        }
    },
    'V08': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '60X',
                    '61X',
                    '62X',
                    '63X'
                ]
            },
            'kjonn': {
                'length': 1,
                'type': 'str',
                'value': 'letter',
                'derived': True,
                'possible_values': [
                    'M'
                ]
            }
        }
    },
    'V46': {
        'variable_values_info': {
            'diagnosedato': {
                'length': 10,
                'type': 'date',
                'derived': False
            },
            'dodsdato': {
                'length': 10,
                'type': 'date',
                'value': '',
                'derived': False
            }
        }
    },
    'V03': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '50X'
                ]
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    '0',
                    'A',
                    'B',
                    'C',
                    'D',
                    '9'
                ]
            }
        }
    },
    'V06': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420'
                ]
            },
            'meldingstype': {
                'length': 1,
                'type': 'str',
                'value': 'letter',
                'derived': True,
                'possible_values': [
                    'H'
                ]
            }
        }
    },
    'V12': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '509'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1708',
                    'XXXX'
                ]
            },
            'kirurgi': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '11'
                ]
            }
        }
    },
    'V27': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '50X'
                ]
            },
            'kirurgi': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '10',
                    'XX'
                ]
            },
            'kjonn': {
                'length': 1,
                'type': 'str',
                'value': 'letter',
                'derived': True,
                'possible_values': [
                    'M'
                ]
            }
        }
    },
    'V10': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '33',
                    '38',
                    '45',
                    '47'
                ]
            }
        }
    },
    'V30': {
        'variable_values_info': {
            'kirurgi': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    'Not',
                    '16',
                    '19'
                ]
            }
        }
    },
    'V32': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': False
            },
            'patologiStadium': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V50': {
        'variable_values_info': {
            'diagnosedato': {
                'length': 10,
                'type': 'date',
                'derived': False
            },
            'fodselsdato': {
                'length': 10,
                'type': 'date',
                'derived': False
            }
        }
    },
    'V54': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'ct': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V36': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '809'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '33'
                ]
            }
        }
    },
    'V55': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'pt': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V37': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '206X',
                    '207X'
                ]
            },
            'metastase': {
                'length': 1,
                'type': 'str',
                'value': 'numberORletter',
                'derived': True,
                'possible_values': [
                    '9'
                ]
            }
        }
    },
    'V56': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'cn': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V57': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'cm': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V41': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '809'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '206X',
                    '207X'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '70',
                    '74',
                    '75',
                    '76'
                ]
            }
        }
    },
    'V58': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'pt': {
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
            }
        }
    },
    'V59': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'pn': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V48': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1708',
                    '1573',
                    '1559',
                    '1770',
                    '1801',
                    '1802',
                    '1808',
                    '1810',
                    '1811',
                    '1813',
                    '1814',
                    '1818',
                    '1819',
                    '1994',
                    '1930',
                    '1931',
                    '1932',
                    '1933',
                    '1934',
                    '1937',
                    '1938',
                    '1939',
                    '1953',
                    '1954',
                    '1955',
                    '1705',
                    '1535',
                    '1545',
                    '1550',
                    '181X',
                    '194X',
                    '206X',
                    '207X',
                    '190X'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '98',
                ]
            }
        }
    },
    'V60': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
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
            }
        }
    },
    'V61': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'pm': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V62': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
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
            }
        }
    },
    'V63': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '206X',
                    '207X'
                ]
            },
            'kliniskStadium': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V65': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1708',
                    '1573',
                    '1559',
                    '1770',
                    '1801',
                    '1802',
                    '1808',
                    '1810',
                    '1811',
                    '1813',
                    '1814',
                    '1818',
                    '1819',
                    '1994',
                    '1930',
                    '1931',
                    '1932',
                    '1933',
                    '1934',
                    '1937',
                    '1938',
                    '1939',
                    '1953',
                    '1954',
                    '1955',
                    '1705',
                    '1535',
                    '1545',
                    '1550',
                    '181X',
                    '194X',
                    '206X',
                    '207X',
                    '190X'
                ]
            },
            'side': {
                'length': 5,
                'type': 'str',
                'value': '',
                'derived': True,
                'possible_values': ''
            }
        }
    },
    'V70': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '190X'
                ]
            },
            'ekstralokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '8888'
                ]
            },
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '935019',
                    '9990XX',
                    '815XXX',
                    '824XXX',
                    '868XXX',
                    '869XXX',
                    '8700XXX',
                    'XXXX3X',
                    'XXXX0X',
                    'XXXX1X',
                    'XXXX2X',
                    '81903X',
                    '85503X',
                    '872XXX',
                    '873XXX',
                    '874XXX',
                    '875XXX',
                    '876XXX',
                    '877XXX',
                    '878XXX',
                    '879XXX',
                    '8970XX'
                ]
            },
            'diagnosedato': {
                'length': 10,
                'type': 'date',
                'derived': False
            },
            'pm': {
                'length': 10,
                'type': 'str',
                'value': '',
                'derived': False
            }
        }
    },
    'V18': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '420',
                    '421',
                    '76X',
                    '752',
                    '809',
                    '220',
                    '50X',
                    '60X',
                    '61X',
                    '62X',
                    '63X',
                    '67X',
                    '73X',
                    '25X',
                    '34X',
                    '48X',
                    '51X',
                    '52X',
                    '53X',
                    '54X',
                    '55X',
                    '56X',
                    '57X'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1573'
                ]
            },
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '815XXX',
                    '824XXX',
                    '868XXX',
                    '869XXX',
                    '870XXX'
                ]
            }
        }
    },
    'V39': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '220'
                ]
            },
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1559'
                ]
            },
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '81903X',
                    '85503X'
                ]
            }
        }
    },
    'V68': {
        'variable_values_info': {
            'lokalisasjon': {
                'length': 4,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '1550'
                ]
            },
            'morfologi': {
                'length': 6,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '817XXX',
                    '818XXX',
                    '8970XX'
                ]
            }
        }
    },
    'V69': {
        'variable_values_info': {
            'topografi': {
                'length': 3,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '619'
                ]
            },
            'basis': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': ['57', '60', '79', '98']
            },
            'kirurgi': {
                'length': 2,
                'type': 'str',
                'value': 'number',
                'derived': True,
                'possible_values': [
                    '97'
                ]
            },
            'meldingstype': {
                'length': 1,
                'type': 'str',
                'value': 'letter',
                'derived': True,
                'possible_values': [
                    'H'
                ]
            }
        }
    }
}
