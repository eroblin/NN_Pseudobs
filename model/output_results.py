'''
Output the results based on the model previously trained
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sklearn.utils import resample
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv as skSurv

def determine_surv_prob(surv,t):
    """ Compute the survival probability at a specific time point
    # Arguments
        surv: matrix of predictions at every event time (in lines) for each individual (in columns). Times are in index.
        t: time of survival prediction
    # Returns
        result: prediction of survival at time t    
    """
    for (i,t2) in enumerate(surv.index):
        if (t2>t):
            if (i==0):
                result = (t/t2)*surv.iloc[i]+ (t2-t)/t2*1
                break
            else :
                result = (t-surv.index.values[i-1])/(t2-surv.index.values[i-1])*surv.iloc[i]+(t2-t)/(t2-surv.index.values[i-1])*surv.iloc[i-1]
                break
        else :
            result = surv.iloc[i]
    return(result)


def output_stats(model,surv,X_train, df_train, X_val, df_val):
    """ Compute the output of the model on the test set
    # Arguments
        model: neural network model trained with final parameters.
        X_train : input variables of the training set
        df_train: training dataset
        X_val : input variables of the validation set
        df_val: validation dataset
    # Returns
        results_test: Uno C-index at 5 and 10 years and Integrated Brier Score
    """ 
    time_grid = np.linspace(np.percentile(df_val['yy'],10), np.percentile(df_val['yy'],90),100)
    data_train = skSurv.from_arrays(event=df_train['status'], time=df_train['yy'])
    data_test = skSurv.from_arrays(event=df_val['status'], time=df_val['yy'])
    c5 =  concordance_index_ipcw(data_train, data_test, np.array(-determine_surv_prob(surv,5)), 5)[0]
    c10 = concordance_index_ipcw(data_train, data_test, np.array(-determine_surv_prob(surv,10)), 10)[0]
    ev = EvalSurv(surv, np.array(df_val['yy']), np.array(df_val['status']), censor_surv='km')
    ibs = ev.integrated_brier_score(time_grid)
    res = pd.DataFrame([c5,c10,ibs]).T
    res.columns = ['unoc5', 'unoc10','ibs']
    return res


def output_sim_data(model,surv,X_train, df_train, X_test, df_test):
    """ Compute the output of the model on the test set
    # Arguments
        model: neural network model trained with final parameters.
        X_train : input variables of the training set
        df_train: training dataset
        X_val : input variables of the validation set
        df_val: validation dataset
    # Returns
        results_test: Uno C-index at median survival time and Integrated Brier Score
    """ 
    time_grid = np.linspace(np.percentile(df_test['yy'],10), np.percentile(df_test['yy'],90),100)
    median_time = np.percentile(df_test['yy'],50)
    data_train = skSurv.from_arrays(event=df_train['status'], time=df_train['yy'])
    data_test = skSurv.from_arrays(event=df_test['status'], time=df_test['yy'])
    
    c_med =  concordance_index_ipcw(data_train, data_test, np.array(-determine_surv_prob(surv,median_time)), median_time)[0]
    ev = EvalSurv(surv, np.array(df_test['yy']), np.array(df_test['status']), censor_surv='km')
    ibs = ev.integrated_brier_score(time_grid)
    res = pd.DataFrame([c_med,ibs]).T
    res.columns = ['c_median', 'ibs']
    return res


