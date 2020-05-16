#from nilmtk.electric import get_activations
import pandas as pd
import numpy as np
from disaggregate import get_activations

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, recall_score, precision_score, accuracy_score

class Metrics():
    def __init__(self, y_true, y_pred, params,isState):
        self.params = params
        self.s_true = pd.DataFrame(np.zeros_like(y_true), index=y_true.index)
        self.s_pred = pd.DataFrame(np.zeros_like(y_pred), index=y_pred.index)
        if(isState):
            self.e_true = y_true
            self.e_pred = y_pred
            self.s_true = y_true
            self.s_pred[y_pred > 0.5]=1.0   # may be wrong, np.int8?
        else:
            self.e_true = y_true
            self.e_pred = y_pred

            self.calculate_state()

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.precision = 0
        self.recall = 0
        self.true_on_period = float(len(np.where(self.s_true==1)[0]))
        self.true_off_period = float(len(np.where(self.s_true==0)[0]))
        self.pred_on_period = float(len(np.where(self.s_pred==1)[0]))
        self.pred_off_period = float(len(np.where(self.s_pred==0)[0]))
        self.calculate_cf_matrix()


    def calculate_state(self):
        _, self.s_true = get_activations(self.e_true, self.params)
        _, self.s_pred = get_activations(self.e_pred, self.params)

    def calculate_cf_matrix(self):
        temp = confusion_matrix(self.s_true, self.s_pred)
        print(temp)
        self.TP = temp[1][1]
        self.TN = temp[0][0]
        self.FP = temp[0][1]
        self.FN = temp[1][0]

    def Accuracy(self):
        return accuracy_score(self.s_true, self.s_pred)

    def Precision(self):
        if (self.TP+self.FP != 0):
            p = self.TP/(self.TP+self.FP)
            self.precision = p
        else:
            p = 0
        return p

    def Recall(self):
        if (self.TP+self.FN != 0):
            r = self.TP/(self.TP+self.FN)
            self.recall = r
        else:
            r = 0
        return r

    def F_1_score(self):
        if(self.precision == 0 or self.recall == 0):
            return 0
        else:
            return f1_score(self.s_true, self.s_pred)

    def MSE(self):
        return mean_squared_error(self.e_true, self.e_pred)
    
    def MAE(self):
        return mean_absolute_error(self.e_true, self.e_pred)

    def sMAE(self,rate=100.0):
        error = np.array((self.e_true - self.e_pred)).flatten()
        abs_error = np.abs(error)
        s = np.array(self.s_true).flatten()
        e1 = sum(abs_error * s) / self.true_on_period
        e2 = sum(abs_error * (1 - s)) / self.true_off_period
        return (e1 * rate + e2) / (1 + rate)



    # def RMSE(self):
    #     '''
    #     The root mean square error (Chris Holmes, 2014;Batra, Kelly, et al., 2014 ; Mayhorn et al., 2016) 
    #     is the standard deviation of the energy estimation errors. The RMSE reports based on how spread-out 
    #     these errors are. In other words, it tells you how concentrated the estimations are around the true 
    #     values. The RMSE reports on the same unit as the data, thus making it an intuitive metric.
    #     '''
    #     MSE = self.MSE()
    #     E_mean = np.mean(self.e_true)
    #     return 1 - np.square(MSE)/E_mean

        



# if __name__ = '__main__':
#     from nilmtk import DataSet
#     ukdale = DataSet(r'D:\workspace\data\ukdale.h5')
#     elec = ukdale.buildings[1].elec
#     print(elec)
#     elec_series = elec[2].power_series_all_data(sample_period=1).head(10800)
#     # state = get_activations_2(elec_series,25,300,300)
#     # print(state)
    

        

        
    







# activation = get_activations(elec_series)
