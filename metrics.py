#from nilmtk.electric import get_activations
import pandas as pd
import numpy as np
from disaggregate import get_activations

# def timedelta64_to_secs(timedelta):
#     """Convert `timedelta` to seconds.

#     Parameters
#     ----------
#     timedelta : np.timedelta64

#     Returns
#     -------
#     float : seconds
#     """
#     if len(timedelta) == 0:
#         return np.array([])
#     else:
#         return timedelta / np.timedelta64(1, 's')

# def get_activations(chunk, min_off_duration=10, min_on_duration=0,
#                     border=1, on_power_threshold=5):
#     """Returns runs of an appliance.

#     Most appliances spend a lot of their time off.  This function finds
#     periods when the appliance is on.

#     Parameters
#     ----------
#     chunk : pd.Series
#     min_off_duration : int
#         If min_off_duration > 0 then ignore 'off' periods less than
#         min_off_duration seconds of sub-threshold power consumption
#         (e.g. a washing machine might draw no power for a short
#         period while the clothes soak.)  Defaults to 0.
#     min_on_duration : int
#         Any activation lasting less seconds than min_on_duration will be
#         ignored.  Defaults to 0.
#     border : int
#         Number of rows to include before and after the detected activation
#     on_power_threshold : int or float
#         Watts

#     Returns
#     -------
#     list of pd.Series.  Each series contains one activation.
#     """
#     when_on = chunk >= on_power_threshold
#     # print(chunk)
#     state = pd.DataFrame(np.zeros_like(chunk), index=chunk.index)
#     # print(state)
#     # Find state changes
#     state_changes = when_on.astype(np.int8).diff()

#     switch_on_events = np.where(state_changes == 1)[0]
#     switch_off_events = np.where(state_changes == -1)[0]


#     if len(switch_on_events) == 0 or len(switch_off_events) == 0:
#         if(when_on[0]):
#             state[:] = 1
#             return [], state
#         else:
#             return [], state

#     del when_on
#     del state_changes

#     # Make sure events align
#     if switch_off_events[0] < switch_on_events[0]:
#         state[:switch_off_events[0]] = 1
#         switch_off_events = switch_off_events[1:]
#         if len(switch_off_events) == 0:
#             return [], state
#     if switch_on_events[-1] > switch_off_events[-1]:
#         state[switch_on_events[-1]:] = 1
#         switch_on_events = switch_on_events[:-1]
#         if len(switch_on_events) == 0:
#             return [], state
#     assert len(switch_on_events) == len(switch_off_events)

#     # Smooth over off-durations less than min_off_duration
#     if min_off_duration > 0:
#         off_durations = (chunk.index[switch_on_events[1:]].values -
#                          chunk.index[switch_off_events[:-1]].values)

#         off_durations = timedelta64_to_secs(off_durations)

#         above_threshold_off_durations = np.where(
#             off_durations >= min_off_duration)[0]

#         # Now remove off_events and on_events
#         switch_off_events = switch_off_events[
#             np.concatenate([above_threshold_off_durations,
#                             [len(switch_off_events)-1]])]
#         switch_on_events = switch_on_events[
#             np.concatenate([[0], above_threshold_off_durations+1])]
#     assert len(switch_on_events) == len(switch_off_events)

#     activations = []
#     for on, off in zip(switch_on_events, switch_off_events):
#         duration = (chunk.index[off] - chunk.index[on]).total_seconds()
#         if duration < min_on_duration:
#             continue
#         on -= 1 + border
#         if on < 0:
#             on = 0
#         off += border
#         activation = chunk.iloc[on:off]
#         state.iloc[on:off] = 1
#         # throw away any activation with any NaN values
#         if not activation.isnull().values.any():
#             activations.append(activation)

#     return activations, state



# # def get_act_2(chunk, power_on, power_off, N_on, N_of):
# #     assert power_on == power_off
# #     when_on = chunk >= power_on
# #     state_change = when_on.astype(np.int8).diff()
# def get_activations_2(chunk,p,N_on,N_off,sample_period):
#     N_on = int(N_on/sample_period)
#     N_off = int(N_off/sample_period)
#     state = pd.Series(np.zeros_like(chunk),index = chunk.index)
#     transform = []
#     def stay(i,L,state,flag):
#         state[i] = state[i-1]
#         if(len(L) != 0):
#             state[L] = flag
#         L.clear()

#     def change(i,L,state,N,flag):
#         L.append(state.index[i])
#         if(len(L)>=N):
#             state[L] = flag
#             L.clear()

#     for i in range(1,len(chunk)):
#         #print(state)
#         if(chunk[i] < p) and (state[i-1] == 0):
#             stay(i,transform,state,0)
#         elif(chunk[i] > p and state[i-1] == 1):
#             stay(i,transform,state,1)
#         elif(chunk[i] < p and state[i-1] == 1):
#             change(i,transform,state,N_off,0)
#         elif(chunk[i] > p and state[i-1] == 0):
#             change(i,transform,state,N_on,1)
#     return state




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



        return error1

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
