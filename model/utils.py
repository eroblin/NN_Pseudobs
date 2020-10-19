'''
Functions to prepare the data
'''

import numpy as np
import pandas as pd
from pycox.models import CoxTime
from pycox.models import DeepHitSingle
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

def prepare_data(df_train, df_test,name):
    """ Define the input and output sets formated to use for neural network model 
    # Arguments
        df_train: training set with all input variables, survival time and censoring status
        df_test: test set with all input variables, survival time and censoring status
        name: name of the model (CoxCC, CoxTime or DeepHit)
    # Returns
        x_train: input variables for the training set
        y_train: output variables for the training set
        x_test: input variables for the test set
        duration_test: survival time for the test set
        event_test: censoring indicator for the test set
        labtrans: output variables transformed for specific models (DeepHit ad CoxTime)
    """
    col_list = list(df_train.columns)
    cols_standardize = [e for e in col_list if e not in ['yy', 'status']]
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    x_mapper = DataFrameMapper(standardize)
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')
    get_target = lambda df: (df['yy'].values, df['status'].values)
    
    if name=="DeepHit" :
        num_durations = 10
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_train = labtrans.fit_transform(*get_target(df_train))
    elif name=="CoxTime":
        labtrans = CoxTime.label_transform()
        y_train = labtrans.fit_transform(*get_target(df_train))
    else :
        labtrans = ""
        y_train = get_target(df_train)
    duration_test, event_test = get_target(df_test)
    
    return x_train, y_train, x_test, duration_test, event_test,labtrans

def prepare_pseudobs_simu(df_train, y_train, df_test,name):
    """ Prepare the data for training
    The input data is formated so that one line corresponds to one subject at a particular time point.
    # Arguments
        df_train: the entire dataset (input + survival times + event status)
        y_train: the pseudo-values computed according to the method chosen. 
        df_test: the entire dataset (input + survival times + event status)
    # Returns
        x_train_all: input data with all input variables + time variable and one line represents one subject at one time point.
        y_train_all: pseudo-values computed according to the method chosen. 
        x_test_all: input data with all input variables + time variable and one line represents one subject at one time point.
        y_test_all: survival time and event status.
        n_picktime: the number of time point at which the pseudo-observations are computed.
    """
    y_test_all = df_test[['yy','status']]
    n_picktime = int(y_train[['s']].apply(pd.Series.nunique))
    x_test = df_test.drop(['yy','status'], axis = 1)
    x_test_all = pd.concat([x_test]*n_picktime)
    time_test = pd.DataFrame(np.repeat(np.unique(y_train[['s']]),len(x_test)))
    x_test_all.reset_index(inplace=True, drop=True)
    x_test_all = pd.concat([x_test_all, time_test], axis = 1)

    if name!= "pseudo_discrete":
        x_train = df_train.drop(['yy','status'], axis = 1)
        x_train_all = pd.concat([x_train]*n_picktime)
        x_train_all.reset_index(inplace=True, drop=True)
        x_train_all = pd.concat([x_train_all, y_train[['s']]], axis = 1)
        y_train_all = y_train[['pseudost']]
    else:
        x_train = df_train.drop(['yy','status'], axis = 1)
        x_train['id'] = np.arange(len(x_train)) + 1
        x_train = x_train.merge(y_train, left_on='id', right_on='id')
        x_train_all = x_train.drop(['id','pseudost'], axis = 1)
        y_train_all = x_train['pseudost']
    # Data normalization
    col_list = list(x_train_all.columns)
    x_test_all.columns = col_list
    cols_standardize = [e for e in col_list]
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    x_mapper = DataFrameMapper(standardize, df_out=True)
    x_train_all = x_mapper.fit_transform(x_train_all).astype('float32')
    x_test_all = x_mapper.transform(x_test_all).astype('float32')
    
    return(x_train_all, y_train_all, x_test_all, y_test_all, n_picktime)

def prepare_pseudobs_metabric(y,df, pseudo_type):
    """ Define the input and output sets formated to use for neural network model with pseudoobservations
    # Arguments
        y: pseudo observations
        df: Dataset with all input variables, survival time and censoring status
        pseudo_type: type of pseudo observation (pseudo_discrete, pseudo_optim, pseudo_continuous, pseudo_km)
    # Returns
        x_all : input variables (one line for one patient at one timepoint)
        y_all : pseudoobservations (one line for one patient at one timepoint)
    """
    
    if pseudo_type != "pseudo_discrete" and pseudo_type != "pseudo_optim2":
        n_picktime = int(y[['s']].apply(pd.Series.nunique))
        x = df.drop(['yy','status'], axis = 1)
        x_all = pd.concat([x]*n_picktime)
        x_all.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        x_all = pd.concat([x_all, y[['s']]], axis = 1)
        y_all = y[['pseudost']]
    else : 
        x = df.drop(['yy','status'], axis = 1)
        x = x.merge(y, left_on='id', right_on='id')
        x_all = x.drop(['pseudost'], axis = 1)
        y_all = x['pseudost']
    return x_all, y_all