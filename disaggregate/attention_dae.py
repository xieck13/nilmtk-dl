#! -*- coding: utf-8 -*-
#%%
from __future__ import print_function, division
from keras import backend as K
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten, add, MaxPooling1D, Input, UpSampling1D, BatchNormalization, GaussianNoise, Activation
from keras.regularizers import l2

class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size #必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                 2 * K.arange(self.size / 2, dtype='float32' \
                               ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 #K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def resnet_layer(inputs,
                 filters=16,
                 kernel_size=3,
                 strides=1,
                 batch_normalization=True,
                 conv_first=True,
                 dilation_rate=7,
                 std = 1e-4):
    conv = Conv1D(filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  dilation_rate = dilation_rate)

    x = inputs
    x = conv(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = GaussianNoise(stddev = std)(x)
    return x

def coder(input,
          filters,
          kernel_size = 3,
          down_sample = True,
          size = 2):
    input = resnet_layer(input,filters=filters,kernel_size=kernel_size)
    x = resnet_layer(input,filters=filters,kernel_size=kernel_size)
    input = add([input, x])
    x = resnet_layer(input,filters=filters,kernel_size=kernel_size)
    x = add([input, x])
    if(down_sample):
        x = MaxPooling1D(pool_size=size)(x)
    else:
        x = UpSampling1D(size=size)(x)
    return x

def begin(input, std=1e-5):
    x = Conv1D(filters=64, kernel_size=1, activation='sigmoid',padding='same')(input)
    x = BatchNormalization()(x)
    x = GaussianNoise(std)(x)
    return x

def end(input):
    x = Conv1D(filters=1, kernel_size=3, activation='sigmoid', dilation_rate=7,padding='same')(input)
    x = BatchNormalization()(x)
    return x

def midden(input):
    x = Attention(8,32)([input,input,input])
    x = add([x,input])
    return x

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
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def generate_slide_window(arr, step=1, n=1800):
    i = 0
    result = []
    while(i < (len(arr)-n+1)):
        result.append(arr[i:i+n])
        i+=step
    return np.array(result)






# %%





from warnings import warn
from nilmtk.disaggregate import Disaggregator
from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten, add, MaxPooling1D, Input
import keras
import pandas as pd
import numpy as np
from collections import OrderedDict 
from keras.optimizers import SGD
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import keras.backend as K
from statistics import mean
import os
import pickle
import random
import json
from .util import *



random.seed(10)
np.random.seed(10)

class ADAE(Disaggregator):
    
    def __init__(self, params):
        """
        Iniititalize the moel with the given parameters
        """
        self.MODEL_NAME = "ADAE"
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',1800)
        self.stride_length = params.get('stride_length',self.sequence_length)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size',16)
        self.mains_mean = params.get('mains_mean',1000)
        self.mains_std = params.get('mains_std',600)
        self.appliance_params = params.get('appliance_params',{})
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.models = OrderedDict()
        if self.load_model_path:
            self.load_model()
        

        
    def partial_fit(self, train_main, train_appliances, do_preprocessing=True,**load_kwargs):        
        """
        The partial fit function
        """

        # If no appliance wise parameters are specified, then they are computed from the data
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        # TO preprocess the data and bring it to a valid shape
        if do_preprocessing:
            print ("Doing Preprocessing")
            train_main,train_appliances = self.call_preprocessing(train_main,train_appliances,'train')
        train_main = pd.concat(train_main,axis=0).values
        train_main = train_main.reshape((-1,self.sequence_length,1))
        new_train_appliances  = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df,axis=0).values
            app_df = app_df.reshape((-1,self.sequence_length,1))
            new_train_appliances.append((app_name, app_df))
        train_appliances = new_train_appliances
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print ("First model training for ",appliance_name)
                self.models[appliance_name] = self.return_network()
                print (self.models[appliance_name].summary())
            print ("Started Retraining model for ",appliance_name)    
            model = self.models[appliance_name]
            filepath = 'adae-temp-weights-'+str(random.randint(0,100000))+'.h5'
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            train_x,v_x,train_y,v_y = train_test_split(train_main,power,test_size=.15,random_state=10)  
            def data_generator(data, targets, batch_size):
                batches = (len(data) + batch_size - 1)//batch_size
                while(True):
                    for i in range(batches):
                        X = data[i*batch_size : (i+1)*batch_size]
                        Y = targets[i*batch_size : (i+1)*batch_size]
                        yield (X, Y)

            lr_scheduler = LearningRateScheduler(lr_schedule)

            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                           cooldown=0,
                                           patience=3,
                                           min_lr=0.5e-6)
                         
            model.fit_generator(generator = data_generator(train_x, train_y, self.batch_size),
                                steps_per_epoch = (len(train_x) + self.batch_size - 1) // self.batch_size,
                                epochs = self.n_epochs,
                                verbose = 1,
                                callbacks = [checkpoint,lr_scheduler,lr_reducer],
                                validation_data = (v_x, v_y)
            )
#           model.fit(train_x,train_y,validation_data = [v_x,v_y],epochs = self.n_epochs, callbacks = [checkpoint,lr_scheduler,lr_reducer],shuffle=True,batch_size=self.batch_size)
            model.load_weights(filepath)

        if self.save_model_path:
            self.save_model()

    def load_model(self):
        print ("Loading the model using the pretrained-weights")        
        model_folder = self.load_model_path
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
        
        if os.path.exists(self.save_model_path) == False:
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



    def disaggregate_chunk(self, test_main_list, do_preprocessing=True):
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list,submeters_lst=None,method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1,self.sequence_length,1))
            disggregation_dict = {}
            for appliance in self.models:
                prediction = self.models[appliance].predict(test_main,batch_size=self.batch_size)
                app_mean = self.appliance_params[appliance]['mean']
                app_std = self.appliance_params[appliance]['std']
                prediction = self.denormalize_output(prediction,app_mean,app_std)
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions>0,valid_predictions,0)
                series = pd.Series(valid_predictions)
                disggregation_dict[appliance] = series
            results = pd.DataFrame(disggregation_dict,dtype='float32')
            test_predictions.append(results)
        return test_predictions
            
    def return_network(self):

        inputs = Input(shape=(1800,1), dtype='float32')
        x = begin(inputs)
        x = coder(x, filters=128)
        x = coder(x, filters=256)
        x = midden(x)
        x = midden(x)
        x = coder(x, filters=256, down_sample=False)
        x = coder(x, filters=128, down_sample=False)
        outputs = end(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer='Adam')
        print(model)

        return model
    

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        sequence_length  = self.sequence_length
        if method=='train':
            processed_mains = []
            for mains in mains_lst:                
                mains = self.normalize_input(mains.values,sequence_length,self.mains_mean,self.mains_std, True)
                processed_mains.append(pd.DataFrame(mains))

            tuples_of_appliances = []
            for (appliance_name,app_df_list) in submeters_lst:
                app_mean = self.appliance_params[appliance_name]['mean']
                app_std = self.appliance_params[appliance_name]['std']
                processed_app_dfs = []
                for app_df in app_df_list:
                    data = self.normalize_output(app_df.values, sequence_length,app_mean,app_std, True)
                    processed_app_dfs.append(pd.DataFrame(data))                    
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances

        if method=='test':
            processed_mains = []
            for mains in mains_lst:                
                mains = self.normalize_input(mains.values,sequence_length,self.mains_mean,self.mains_std,False)
                processed_mains.append(pd.DataFrame(mains))
            return processed_mains
    
        
    def normalize_input(self,data,sequence_length, mean, std, overlapping=False):
        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0)   
        if overlapping:
            # windowed_x = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
            windowed_x = generate_slide_window(arr, self.stride_length, n)
        else:
            windowed_x = arr.reshape((-1,sequence_length))
        windowed_x = windowed_x - mean
        windowed_x = windowed_x/std
        return (windowed_x/std).reshape((-1,sequence_length))

    def normalize_output(self,data,sequence_length, mean, std, overlapping=False):
        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0) 
        if overlapping:  
            # windowed_y = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
            windowed_y = generate_slide_window(arr, self.stride_length, n)
        else:
            windowed_y = arr.reshape((-1,sequence_length))        
        windowed_y = windowed_y - mean
        return (windowed_y/std).reshape((-1,sequence_length))

    def denormalize_output(self,data,mean,std):
        return mean + data*std
    
    def set_appliance_params(self,train_appliances):

        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})




# %%





    

# %%
