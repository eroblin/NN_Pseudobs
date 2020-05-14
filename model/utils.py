'''
Functions to prepare the data
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper


def normed_data(df_train, df_test):
    """ Define the structure of the neural network for a Cox-MLP (CC), CoxTime and  DeepHit
    # Arguments
        df_train: Training set of simulated data with 20 entry variables, survival status and survival time. 
        df_test: Test set of simulated data with 20 entry variables, survival status and survival time. 
    # Returns
        x_train: dataframe with the normalized explanatory variables.
        x_test: dataframe with the normalized explanatory variables.
    """
    col_list = list(df_train.columns)
    cols_standardize = [e for e in col_list if e not in ['yy', 'status']]
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    x_mapper = DataFrameMapper(standardize)
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')
    return x_train, x_test

def prepare_pseudobs(x_train, y_train, df_train, x_test, df_test, name):
    """ Define the input and output sets formated to use for neural network model with pseudoobservations
    # Arguments
        x_train: dataframe with the normalized explanatory variables
        y_train: survival time
        df_train: Training set of simulated data with 20 entry variables, survival status and survival time
        x_test: dataframe with the normalized explanatory variables
        df_test: Test set of simulated data with 20 entry variables, survival status and survival time
        name: name of the model
    # Returns
        x_train_all : input variables for the training set (one line for one patient at one timepoint)
        y_train_all : pseudoobservation (one line for one patient at one timepoint)
        x_test_all : input variables for the test set (one line for one patient at one timepoint)
    """
    n_picktime = int(y_train[['s']].apply(pd.Series.nunique))
    x_test_all = pd.concat([pd.DataFrame(x_test)]*n_picktime)
    time_test = pd.DataFrame(np.repeat(np.unique(y_train[['s']]),len(x_test)))
    time_test.columns = ['s']
    x_test_all.reset_index(inplace=True, drop=True)
    x_test_all = pd.concat([x_test_all, time_test], axis = 1)
    x_train = pd.DataFrame(x_train)
    if name != "pseudo-discrete":
        x_train_all = pd.concat([x_train]*n_picktime)
        x_train_all.reset_index(inplace=True, drop=True)
        x_train_all = pd.concat([x_train_all, y_train[['s']]], axis = 1)
        y_train_all = y_train[['pseudost']]
    else:
        x_train['id'] = np.arange(len(x_train)) + 1
        x_train = x_train.merge(y_train, left_on='id', right_on='id')
        x_train_all = x_train.drop(['id','pseudost'], axis = 1)
        y_train_all = x_train['pseudost']
    return x_train_all, y_train_all, x_test_all





