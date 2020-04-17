import os
import warnings

warnings.filterwarnings("ignore")
from nilmtk.dataset import DataSet
from nilmtk.metergroup import MeterGroup
import pandas as pd
from nilmtk.losses import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
from IPython.display import clear_output
from metrics import Metrics
from disaggregate import config, get_activations, get_sections_df, get_sections_df_2
import copy
import joblib


class API():
    """
    The API is designed for rapid experimentation with NILM Algorithms. 
    """

    def __init__(self, params):

        """
        Initializes the API with default parameters
        """
        self.power = {}
        self.sample_period = 1
        self.appliances = []
        self.methods = {}
        self.chunk_size = None
        self.pre_trained = False
        self.metrics = []
        self.train_datasets_dict = {}
        self.test_datasets_dict = {}
        self.artificial_aggregate = False
        self.train_submeters = []
        self.train_mains = pd.DataFrame()
        self.test_submeters = []
        self.test_mains = pd.DataFrame()
        self.test_sections = []
        self.gt_overall = {}
        self.pred_overall = {}
        self.classifiers = []
        self.DROP_ALL_NANS = True
        self.mae = pd.DataFrame()
        self.rmse = pd.DataFrame()
        self.errors = pd.DataFrame()
        self.predictions = []
        self.errors_keys = []
        self.predictions_keys = []
        self.params = params
        for elems in params['power']:
            self.power = params['power']
        self.sample_period = params['sample_rate']
        for elems in params['appliances']:
            self.appliances.append(elems)

        self.pre_trained = ['pre_trained']
        self.train_datasets_dict = params['train']['datasets']
        self.test_datasets_dict = params['test']['datasets']
        # self.metrics = params['test']['metrics']
        self.methods = params['methods']
        self.artificial_aggregate = params.get('artificial_aggregate', self.artificial_aggregate)
        self.activation_profile = params.get('activation_profile', config['threshold'])
        self.isState = params.get('isState', False)
        self.sec_dict = {}
        self.experiment(params)

    def experiment(self, params):
        """
        Calls the Experiments with the specified parameters
        """

        self.store_classifier_instances()
        d = self.train_datasets_dict

        for model_name, clf in self.classifiers:
            # If the model is a neural net, it has an attribute n_epochs, Ex: DAE, Seq2Point
            print("Started training for ", clf.MODEL_NAME)

            # If the model has the filename specified for loading the pretrained model, then we don't need to load training data

            if hasattr(clf, 'load_model_path'):
                if clf.load_model_path:
                    print(clf.MODEL_NAME, " is loading the pretrained model")
                    continue

                print("Joint training for ", clf.MODEL_NAME)
                self.train_jointly(clf, d)

                # if it doesn't support chunk wise training
            else:
                print("Joint training for ", clf.MODEL_NAME)
                self.train_jointly(clf, d)

            print("Finished training for ", clf.MODEL_NAME)
            clear_output()

        d = self.test_datasets_dict

        print("Joint Testing for all algorithms")
        self.test_jointly(d)

    def train_jointly(self, clf, d):

        # This function has a few issues, which should be addressed soon
        print("............... Loading Data for training ...................")
        # store the train_main readings for all buildings
        self.train_mains = []
        self.train_submeters = [[] for i in range(len(self.appliances))]
        for dataset in d:
            print("Loading data for ", dataset, " dataset")
            train = DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:
                print("Loading building ... ", building)
                train.set_window(start=d[dataset]['buildings'][building]['start_time'],
                                 end=d[dataset]['buildings'][building]['end_time'])
                main_meter = train.buildings[building].elec.mains()
                good_sections = train.buildings[building].elec[self.appliances[0]].good_sections()
                main_df = next(main_meter.load(physical_quantity='power', ac_type=self.power['mains'],
                                               sample_period=self.sample_period))
                # train_df = train_df[[list(train_df.columns)[0]]]

                # main_df_list = get_sections_df(main_df, good_sections)  # train_df
                appliance_readings = []

                for appliance_name in self.appliances:
                    app_meter = train.buildings[building].elec[appliance_name]
                    app_df = next(app_meter.load(physical_quantity='power', ac_type=self.power['appliance'],
                                                 sample_period=self.sample_period))
                    # app_df_list = get_sections_df(app_df, good_sections)

                    if building not in self.sec_dict:
                        self.sec_dict[building] = get_sections_df_2(good_sections, app_meter.good_sections())

                    main_df_list = [main_df[sec[0]:sec[1]] for sec in self.sec_dict[building]]
                    app_df_list = [app_df[sec[0]:sec[1]] for sec in self.sec_dict[building]]

                    appliance_readings.append(app_df_list)  # appliance_readings->app_df_list->app_df

                if self.DROP_ALL_NANS:
                    main_df_list, appliance_readings = self.dropna(main_df_list,
                                                                   appliance_readings)  # Ttrain_list: [pd[sec],pd[sec]..]

                if self.artificial_aggregate:
                    print("Creating an Artificial Aggregate")
                    train_df = pd.DataFrame(np.zeros(appliance_readings[0].shape), index=appliance_readings[0].index,
                                            columns=appliance_readings[0].columns)
                    for app_reading in appliance_readings:
                        train_df += app_reading

                print("Train Jointly")

                self.train_mains += main_df_list  # [[sec],[sec]...]]
                train_submeters = appliance_readings.copy()
                for j, appliance_name in enumerate(self.appliances):
                    if self.isState:
                        for i, app_df in enumerate(appliance_readings[j]):
                            _, train_submeters[j][i] = get_activations(app_df, config['threshold'][appliance_name])
                    self.train_submeters[j] += train_submeters[j]

        appliance_readings = []
        for i, appliance_name in enumerate(self.appliances):
            appliance_readings.append((appliance_name, self.train_submeters[i]))

        self.train_submeters = appliance_readings  # [(app_name, [[sec],[sec]...])...]
        clf.partial_fit(self.train_mains, self.train_submeters)

    def test_jointly(self, d):

        # store the test_main readings for all buildings
        for dataset in d:
            print("Loading data for ", dataset, " dataset")
            test = DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:
                self.test_mains = []
                self.test_submeters = [[] for i in range(len(self.appliances))]

                test.set_window(start=d[dataset]['buildings'][building]['start_time'],
                                end=d[dataset]['buildings'][building]['end_time'])
                test_meter = test.buildings[building].elec.mains()
                good_sections = test.buildings[building].elec[self.appliances[0]].good_sections()
                # self.test_sections = good_sections
                main_df = next(test_meter.load(physical_quantity='power', ac_type=self.power['mains'],
                                               sample_period=self.sample_period))

                main_df_list = get_sections_df(main_df, good_sections)  # train_df
                appliance_readings = []

                for appliance_name in self.appliances:
                    app_meter = test.buildings[building].elec[appliance_name]

                    if building not in self.sec_dict:
                        self.sec_dict[building] = get_sections_df_2(good_sections, app_meter.good_sections())

                    app_df = next(app_meter.load(physical_quantity='power', ac_type=self.power['appliance'],
                                                 sample_period=self.sample_period))

                    main_df_list = [main_df[sec[0]:sec[1]] for sec in self.sec_dict[building]]
                    app_df_list = [app_df[sec[0]:sec[1]] for sec in self.sec_dict[building]]
                    appliance_readings.append(app_df_list)

                if self.DROP_ALL_NANS:
                    main_df_list, appliance_readings = self.dropna(main_df_list, appliance_readings)

                if self.artificial_aggregate:
                    print("Creating an Artificial Aggregate")
                    test_mains = pd.DataFrame(np.zeros(appliance_readings[0].shape), index=appliance_readings[0].index,
                                              columns=appliance_readings[0].columns)
                    for app_reading in appliance_readings:
                        test_mains += app_reading

                print("Test Jointly")

                self.test_mains = (main_df_list)
                test_submeters = appliance_readings.copy()

                for j, appliance_name in enumerate(self.appliances):
                    if self.isState:
                        for i, app_df in enumerate(appliance_readings[j]):
                            _, test_submeters[j][i] = get_activations(app_df, config['threshold'][appliance_name])
                    self.test_submeters[j] = (appliance_name, test_submeters[j])

                self.storing_key = str(dataset) + "_" + str(building)
                self.call_predict(self.classifiers, building)

    def dropna(self, mains_list, appliance_readings):
        """
        Drops the missing values in the Mains reading and appliance readings and returns consistent data by copmuting the intersection
        """
        print("Dropping missing values")

        # The below steps are for making sure that data is consistent by doing intersection across appliances
        new_main_list = mains_list.copy()
        new_appliances_list = appliance_readings.copy()
        for j, mains_df in enumerate(mains_list):
            mains_df = mains_df.dropna()
            # if mains_df.shape[0] < 10:
            #     continue
            for i in range(len(appliance_readings)):
                appliance_readings[i][j] = appliance_readings[i][j].dropna()
            ix = mains_df.index
            for app_df in appliance_readings:
                ix = ix.intersection(app_df[j].index)
            new_main_list[j] = mains_df.loc[ix]
            for i, app_df in enumerate(appliance_readings):
                new_appliances_list[i][j] = app_df[j].loc[ix]
        j = 0
        while (j < len(new_main_list)):
            if (new_main_list[j].shape[0] < 10):
                del new_main_list[j]
                for i in range(len(new_appliances_list)):
                    del new_appliances_list[i][j]
            else:
                j += 1
        print('dropna finished')
        return new_main_list, new_appliances_list

    def store_classifier_instances(self):

        """
        This function is reponsible for initializing the models with the specified model parameters
        """
        for name in self.methods:
            try:

                clf = self.methods[name]
                self.classifiers.append((name, clf))

            except Exception as e:
                print("\n\nThe method {model_name} specied does not exist. \n\n".format(model_name=name))
                print(e)

    def call_predict(self, classifiers, building):
        """
        This functions computers the predictions on the self.test_mains using all the trained models and then compares different learn't models using the metrics specified
        """

        pred_overall = {}
        gt_overall = {}
        for name, clf in classifiers:
            gt_overall, pred_overall[name] = self.predict(clf, self.test_mains, self.test_submeters, self.sample_period,
                                                          'Europe/London')

        self.gt_overall = gt_overall
        self.pred_overall = pred_overall
        test_mains = pd.concat(self.test_mains, axis=0)
        if gt_overall.size == 0:
            print("No samples found in ground truth")
            return None

        for i in gt_overall.columns:
            for clf in pred_overall:
                if not os.path.exists('result/' + self.storing_key + '/' + str(i) + '/' + str(clf) + '/section_image'):
                    os.makedirs('result/' + self.storing_key + '/' + str(i) + '/' + str(clf) + '/section_image')
                if not os.path.exists('result/' + self.storing_key + '/' + str(i)):
                    os.makedirs('result/' + self.storing_key + '/' + str(i))
                if not os.path.exists('result/' + self.storing_key + '/' + str(i) + '/' + str(clf) + '/section_df'):
                    os.makedirs('result/' + self.storing_key + '/' + str(i) + '/' + str(clf) + '/section_df')

        print('section_plot:')

        sec_list = self.sec_dict[building]

        for i in gt_overall.columns:
            gt_overall_list = [gt_overall[i][sec[0]:sec[1]] for sec in sec_list]
            # get_sections_df(gt_overall[i], self.test_sections)
            for j, gt in enumerate(gt_overall_list):
                for clf in pred_overall:
                    pred = pred_overall[clf][i]
                    pred_df_list = [pred[sec[0]:sec[1]] for sec in sec_list]
                    plt.figure(figsize=(6, 3))
                    temp_test_main = self.test_mains[j]
                    temp_gt_overall = gt_overall_list[j]
                    temp_pred_df = pred_df_list[j]
                    plt.plot(temp_test_main)
                    plt.plot(temp_gt_overall)
                    plt.plot(temp_pred_df)
                    plt.savefig('result/' + self.storing_key + '/' + str(i) + '/' + str(clf) + '/section_image/' + str(
                        j) + '.png')
                    plt.show()

                    p = plt.figure(figsize=(6, 9))
                    ax1 = p.add_subplot(3, 1, 1)
                    ax1.plot(temp_test_main)
                    plt.title('mains')
                    ax2 = p.add_subplot(3, 1, 2)
                    ax2.plot(temp_gt_overall)
                    plt.title('appliance')
                    ax3 = p.add_subplot(3, 1, 3)
                    plt.title('predict')
                    ax3.plot(temp_pred_df)
                    plt.savefig(
                        'result/' + self.storing_key + '/' + str(i) + '/' + str(clf) + '/section_image/' + '_' + str(
                            j) + '.png')
                    plt.show()

                    temp_result = pd.DataFrame([], index=temp_pred_df.index, columns=['mains', 'gt', 'predict'])
                    temp_result['mains'] = temp_test_main.values
                    temp_result['gt'] = temp_gt_overall
                    temp_result['predict'] = temp_pred_df

                    temp_result.to_csv(
                        'result/' + self.storing_key + '/' + str(i) + '/' + str(clf) + '/section_df/' + str(
                            j) + '.csv')

                    plt.show()

        for i in gt_overall.columns:
            temp_result = copy.deepcopy(config['result'])
            plt.figure()
            if not self.isState:
                plt.plot(test_mains, label='Mains reading')
            plt.plot(gt_overall[i], label='Truth')
            for clf in pred_overall:
                plt.plot(pred_overall[clf][i], label=clf)
            plt.title(i)
            plt.legend()
            plt.savefig('result/' + self.storing_key + '/' + str(i) + '/' + str(clf) + '/' + 'all.png')

            for clf in pred_overall:
                temp_metrics = Metrics(gt_overall[i], pred_overall[clf][i], self.activation_profile[i], self.isState)
                temp_result['MSE'].append(temp_metrics.MSE())
                temp_result['MAE'].append(temp_metrics.MAE())
                temp_result['ACC'].append(temp_metrics.Accuracy())
                temp_result['Precision'].append(temp_metrics.Precision())
                temp_result['Recall'].append(temp_metrics.Recall())
                temp_result['F1'].append(temp_metrics.F_1_score())
                temp_result['sMAE'].append(temp_metrics.sMAE(100.0))
                # temp_df_result = pd.DataFrame(temp_result, index=[0])

            # plot
            for clf in pred_overall:
                plt.figure()
                plt.plot(gt_overall[i], label='Truth')
                plt.plot(pred_overall[clf][i], label=clf)
                plt.legend()
                plt.savefig('result/' + self.storing_key + '/' + str(i) + '/' + str(clf) + '/' + str(clf) + '.png')

            clfs = [clf for clf in pred_overall]
            df_result = pd.DataFrame(temp_result, index=clfs)
            df_result.to_csv('result/' + self.storing_key + '/' + str(i) + '/metrics.csv')
            print(df_result)
            self.errors = df_result

        # for metric in self.metrics:
        #     try:
        #         loss_function = globals()[metric]                
        #     except:
        #         print("Loss function ",metric, " is not supported currently!")
        #         continue

        #     computed_metric={}
        #     for clf_name,clf in classifiers:
        #         computed_metric[clf_name] = self.compute_loss(gt_overall, pred_overall[clf_name], loss_function)
        #     computed_metric = pd.DataFrame(computed_metric)
        #     print("............ " ,metric," ..............")
        #     print(temp_df_result) 
        #     self.errors.append(computed_metric)
        #     self.errors_keys.append(self.storing_key + "_" + metric)

    def predict(self, clf, test_elec, test_submeters, sample_period, timezone):
        """
        Generates predictions on the test dataset using the specified classifier.
        """

        print("Generating predictions for :", clf.MODEL_NAME)
        # "ac_type" varies according to the dataset used. 
        # Make sure to use the correct ac_type before using the default parameters in this code.   

        pred_list = clf.disaggregate_chunk(test_elec)

        # It might not have time stamps sometimes due to neural nets
        # It has the readings for all the appliances

        concat_pred_df = pd.concat(pred_list, axis=0)

        gt = {}
        for meter, data in test_submeters:
            concatenated_df_app = pd.concat(data, axis=0)
            index = concatenated_df_app.index
            gt[meter] = pd.Series(concatenated_df_app.values.flatten(), index=index)

        gt_overall = pd.DataFrame(gt, dtype='float32')
        pred = {}
        for app_name in concat_pred_df.columns:
            app_series_values = concat_pred_df[app_name].values.flatten()
            # Neural nets do extra padding sometimes, to fit, so get rid of extra predictions
            app_series_values = app_series_values[:len(gt_overall[app_name])]
            pred[app_name] = pd.Series(app_series_values, index=gt_overall.index)
        pred_overall = pd.DataFrame(pred, dtype='float32')
        return gt_overall, pred_overall

    # metrics
    def compute_loss(self, gt, clf_pred, loss_function):
        error = {}
        for app_name in gt.columns:
            error[app_name] = loss_function(gt[app_name], clf_pred[app_name])
        return pd.Series(error)
