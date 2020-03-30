import warnings

from api import API
from disaggregate import ADAE, DAE, Seq2Point, Seq2Seq, WindowGRU, RNN, CNN_RNN



warnings.filterwarnings("ignore")

path = 'D:/workspace/nilm/code/databank/redd_data.h5'

debug = True
test = False
if debug:
    method = {
        'DAE': DAE({'save-model-path': 'DAE', 'pretrained-model-path': None, 'n_epochs': 1, 'batch_size': 256}),
        # 'RNN':RNN({'save-model-path':'RNN','pretrained-model-path':None,'n_epochs':1,'batch_size':256}),
        # 'Seq2Point':Seq2Point({'save-model-path':'Seq2Point','pretrained-model-path':None,'n_epochs':1,'batch_size':256}),
        # 'Seq2Seq':Seq2Seq({'save-model-path':'Seq2Seq','pretrained-model-path':None,'n_epochs':1,'batch_size':256}),
        # 'GRU':WindowGRU({'save-model-path':'GRU','pretrained-model-path':None,'n_epochs':1,'batch_size':256}),
    }
else:
    method = {
        'DAE': DAE({'save-model-path': 'DAE', 'pretrained-model-path': None}),
        'RNN': RNN({'save-model-path': 'RNN', 'pretrained-model-path': None}),
        'Seq2Point': Seq2Point({'save-model-path': 'Seq2Point', 'pretrained-model-path': None}),
        'Seq2Seq': Seq2Seq({'save-model-path': 'Seq2Seq', 'pretrained-model-path': None}),
        'GRU': WindowGRU({'save-model-path': 'GRU', 'pretrained-model-path': None}),
    }
if test:
    method = {
        'DAE': DAE({'save-model-path': 'DAE', 'pretrained-model-path': 'DAE'}),
        'RNN': RNN({'save-model-path': 'RNN', 'pretrained-model-path': 'RNN'}),
        'Seq2Point': Seq2Point({'save-model-path': 'Seq2Point', 'pretrained-model-path': 'Seq2Point'}),
        'Seq2Seq': Seq2Seq({'save-model-path': 'Seq2Seq', 'pretrained-model-path': 'Seq2Seq'}),
        'GRU': WindowGRU({'save-model-path': 'GRU', 'pretrained-model-path': 'GRU'}),
    }

ex1 = {

    'power': {
        'mains': ['apparent', 'active'],
        'appliance': ['apparent', 'active']
    },
    'sample_rate': 6,

    'appliances': ['dish washer'],
    'methods': method,
    'isState': False,
    'train': {
        'datasets': {

            'redd': {
                'path': path,
                'buildings': {
                    1: {
                        'start_time': '2011-04-18',
                        'end_time': '2011-05-24'
                    },
                    3: {
                        'start_time': '2011-04-16',
                        'end_time': '2011-05-30'
                    }

                }

            }
        }
    },

    'test': {
        'datasets': {
            'redd': {
                'path': path,
                'buildings': {
                    2: {
                        'start_time': '2011-04-17',
                        'end_time': '2011-05-22'
                    },
                    4: {
                        'start_time': '2011-04-16',
                        'end_time': '2011-06-03'
                    }
                }
            }
        },
    },
}

test = False
ex_train_microwave = {

    'power': {
        'mains': ['apparent', 'active'],
        'appliance': ['apparent', 'active']
    },
    'sample_rate': 6,

    'appliances': ['microwave'],
    'methods': method,
    'isState': False,
    'train': {
        'datasets': {
            'redd': {
                'path': path,
                'buildings': {
                    1: {
                        'start_time': '2011-04-18',
                        'end_time': '2011-05-08'
                    },
                    2: {
                        'start_time': '2011-04-17',
                        'end_time': '2011-05-22'
                    },
                    3: {
                        'start_time': '2011-04-16',
                        'end_time': '2011-05-30'
                    },

                    # 4: {
                    #     'start_time': '2011-04-16',
                    #     'end_time': '2011-06-03'
                    # }

                }
            }
        }
    },

    'test': {
        'datasets': {
            'redd': {
                'path': path,
                'buildings': {
                    1: {
                        'start_time': '2011-05-11',
                        'end_time': '2011-05-24'
                    },
                    2: {
                        'start_time': '2011-04-27',
                        'end_time': '2011-05-22'
                    },
                    3: {
                        'start_time': '2011-05-23',
                        'end_time': '2011-05-30'
                    },

                    # 4: {
                    #     'start_time': '2011-05-23',
                    #     'end_time': '2011-06-03'
                    # }

                }
            }
        }
    }
}

res = API(ex_train_microwave)
