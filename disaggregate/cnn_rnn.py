from __future__ import print_function, division
from warnings import warn
from nilmtk.disaggregate import Disaggregator
from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten, MaxPooling1D, LSTM
import os
import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict
from keras.optimizers import SGD
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
import random
import sys
import json
from .util import *
random.seed(10)
np.random.seed(10)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-5
    if epoch > 30:
        lr *= 0.5e-3
    elif epoch > 20:
        lr *= 1e-3
    elif epoch > 10:
        lr *= 1e-2
    elif epoch > 5:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class CNN_RNN(Disaggregator):

    def __init__(self, params):
        """
        Parameters to be specified for the model
        """

        self.MODEL_NAME = "CNN_RNN"
        self.models = OrderedDict()
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',100)
        self.n_epochs = params.get('n_epochs', 10 )
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_mean = params.get('mains_mean',1800)
        self.mains_std = params.get('mains_std',600)
        self.appliance_params = params.get('appliance_params',{})
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.models = OrderedDict()
        if self.load_model_path:
            self.load_model()        

    def partial_fit(self,train_main,train_appliances,do_preprocessing=True,
            **load_kwargs):

        # If no appliance wise parameters are provided, then copmute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        print("...............Seq2Point partial_fit running...............")
        # Do the pre-processing, such as  windowing and normalizing

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train') #480374,1 -> 480374,99,  480374,1 -> 480374,1

        train_main = pd.concat(train_main,axis=0) #480374,99
        train_main = train_main.values.reshape((-1,self.sequence_length,1))
        
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df,axis=0)
            app_df_values = app_df.values.reshape((-1,1))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            # Check if the appliance was already trained. If not then create a new model for it
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            # Retrain the particular appliance
            else:
                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = 'seq2point-temp-weights-'+str(random.randint(0,100000))+'.h5'
                    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15,random_state=10)
                    lr_scheduler = LearningRateScheduler(lr_schedule)
                    model.fit(train_x,train_y,validation_data=[v_x,v_y],epochs=self.n_epochs,callbacks=[checkpoint, lr_scheduler],batch_size=self.batch_size)
                    model.load_weights(filepath)
        if self.save_model_path:
            self.save_model()

    def load_model(self):
        print ("Loading the model using the pretrained-weights")        
        model_folder = self.load_model_path
        if os.path.exists(os.path.join(model_folder, "model.json")):
            with open(os.path.join(model_folder, "model.json"), "r") as f:
                model_string = f.read().strip()
                params_to_load = json.loads(model_string)


            self.sequence_length = int(params_to_load['sequence_length'])
            self.mains_mean = params_to_load['mains_mean']
            self.mains_std = params_to_load['mains_std']
            self.appliance_params = params_to_load['appliance_params']

            for appliance_name in self.appliance_params:
                self.models[appliance_name] = self.return_network()
                self.models[appliance_name].load_weights(os.path.join(model_folder,appliance_name+".h5"))


    def save_model(self):
        if (os.path.exists(self.save_model_path) == False):
            os.makedirs(self.save_model_path)    
        params_to_save = {}
        params_to_save['appliance_params'] = self.appliance_params
        params_to_save['sequence_length'] = self.sequence_length
        params_to_save['mains_mean'] = self.mains_mean
        params_to_save['mains_std'] = self.mains_std
        for appliance_name in self.models:
            print ("Saving model for ", appliance_name)
            self.models[appliance_name].save_weights(os.path.join(self.save_model_path,appliance_name+".h5"))

        with open(os.path.join(self.save_model_path,'model.json'),'w') as file:
            file.write(json.dumps(params_to_save, cls=NumpyEncoder))

    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):

        if model is not None:
            self.models = model

        # Preprocess the test mains such as windowing and normalizing

        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}
            for appliance in self.models:
                prediction = self.models[appliance].predict(test_main,batch_size=self.batch_size)
                prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def return_network(self):
        # Model architecture
        model = Sequential()
        model.add(Conv1D(32, 3, activation="relu", input_shape=(self.sequence_length, 1),  strides=1))
        model.add(Conv1D(32, 3, activation="relu", strides=1))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(32,return_sequences=False,stateful=False))
        # model.add(Flatten())
        model.add(Dense(1,activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer='adam')  # ,metrics=[self.mse])
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):

        if method == 'train':
            # Preprocessing for the train data
            mains_df_list = []
            for mains in mains_lst:

                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n
                new_mains = np.pad(new_mains,(units_to_pad,0),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n)]) ####################
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))

            appliance_list = []
            for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print ("Parameters for ", app_name ," were not found!")
                    raise ApplianceNotFoundError()

                processed_appliance_dfs = []

                for app_df in app_df_list:
                    new_app_readings = app_df.values.reshape((-1, 1))
                    # This is for choosing windows
                    # new_app_readings = (new_app_readings - app_mean) / app_std  
                    # Return as a list of dataframe
                    processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_appliance_dfs))
            return mains_df_list, appliance_list

        else:
            # Preprocessing for the test data
            mains_df_list = []

            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad+1),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n)]) ####################
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list

    def set_appliance_params(self,train_appliances):
        # Find the parameters using the first
        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
        print (self.appliance_params)