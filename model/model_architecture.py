'''
Objective functions to build a neural network model according to the parameters
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchtuples as tt

from keras import backend 
from keras import optimizers
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from math import pi,cos, floor
from pycox.models import CoxCC
from pycox.models import CoxTime
from pycox.models import DeepHitSingle
from pycox.models.cox_time import MLPVanillaCoxTime
from sklearn.model_selection import KFold




def build_model(x_train, neurons, drop,  activation, lr_opt, optimizer, n_layers, name, labtrans =""):
    """ Define the structure of the neural network for a Cox-MLP (CC), CoxTime and  DeepHit
    # Arguments
        x_train: input data as formated by the function "prepare_data"
        neurons: number of neurons per hidden layer in the neural network
        drop: dropout rate applied after each hidden layer
        activation: activation function applied after each hidden layer
        lr_opt: learning rate chosen for optimization
        optimizer: optimization algorithm 
        n_layers: number of hidden layers 
        name: name of the model
        labtrans: transformed input variables, including the time variable
    # Returns
        model: pycox model (based on pytorch) with the architecture defined previously
        callbacks: callbacks function   
    """
    in_features = x_train.shape[1]
    if labtrans !="":
        out_features = labtrans.out_features
    else:
        out_features = 1
    nb_neurons = [neurons]*n_layers
    
    if optimizer  == "RMSprop" :
        optim = tt.optim.RMSprop()
        callbacks = [tt.callbacks.Callback()]
        
    elif optimizer == "Adam" : 
        optim = tt.optim.Adam()
        callbacks = [tt.callbacks.Callback()]

    elif optimizer == "Adam_AMSGrad" : 
        optim = tt.optim.Adam(amsgrad = True)
        callbacks = [tt.callbacks.Callback()]
        
    elif optimizer == "SGDWR":
        optim = tt.optim.SGD(momentum=0.9)
        callbacks = [tt.callbacks.LRCosineAnnealing()]

    if activation == 'ReLu':
        act = torch.nn.ReLU
    elif activation == 'elu':
        act = torch.nn.ELU
    elif activation == 'Tanh':
        act = torch.nn.Tanh

    if name == "CoxCC":
        net = tt.practical.MLPVanilla(in_features, nb_neurons, out_features, batch_norm = True,
                                  dropout = drop, activation = act, output_bias=False)
        model = CoxCC(net, optim)
        
    elif name == "CoxTime":
        net = MLPVanillaCoxTime(in_features, nb_neurons, batch_norm = True, dropout = drop, activation = act)
        model = CoxTime(net, optim, labtrans = labtrans)
        
    elif name == "DeepHit":    
        net = tt.practical.MLPVanilla(in_features, nb_neurons, out_features, batch_norm = True,
                              dropout = drop, activation = act, output_bias=False)
        model = DeepHitSingle(net, optim,alpha = 0.2, sigma = 0.1, duration_index = labtrans.cuts)
        
    model.optimizer.set_lr(lr_opt)
    
    return model, callbacks


class CosineAnnealingLearningRateSchedule(Callback):
    # constructor
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()
 
    # calculate learning rate for an epoch
    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)
 
    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs=None):
        # calculate learning rate
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        # set learning rate
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)
        

def build_model_pseudobs(x_train, neurons, drop,  activation, lr_opt, optimizer, n_layers,n_epochs):
    """ Define the structure of the neural network for models with pseudo-observations (optim, continuous, discrete, km)
    # Arguments
        x_train: input data as formated by the function "prepare_data"
        neurons: number of neurons per hidden layer in the neural network
        drop: dropout rate applied after each hidden layer
        activation: activation function applied after each hidden layer
        lr_opt: learning rate chosen for optimization
        optimizer: optimization algorithm 
        n_layers: number of hidden layers 
        n_epochs: number of epochs used for training the model
    # Returns
        model: keras model with the architecture defined previously
        callbacks: callbacks function   
    """
    in_features = x_train.shape[1]

    model = Sequential()
    model.add(Dense(neurons, input_dim=in_features, activation=activation))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(Dropout(rate = drop))
    
    if n_layers > 1 :
        model.add(Dense(neurons, activation=activation))
        model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
        model.add(Dropout(rate = drop))
        
    if n_layers == 3 : 
        model.add(Dense(neurons, activation=activation))
        model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
        model.add(Dropout(rate = drop))
        
    model.add(Dense(1, activation='sigmoid'))
    
    if optimizer == "RMSprop":
        optim = optimizers.RMSprop(learning_rate= lr_opt, rho=0.9)
        callbacks = [Callback()]

    elif optimizer == "Adam":
        optim = optimizers.Adam(learning_rate= lr_opt, beta_1=0.9, beta_2=0.999, amsgrad=False)
        callbacks = [Callback()]

    elif optimizer == "Adam_AMSGrad":
        optim = optimizers.Adam(learning_rate= lr_opt, beta_1=0.9, beta_2=0.999, amsgrad=True)
        callbacks = [Callback()]

    elif optimizer == "SGDWR":
        optim = optimizers.SGD(momentum=0.9)
        n_cycles = n_epochs / 50
        callbacks = [CosineAnnealingLearningRateSchedule(n_epochs, n_cycles, 0.01)]
    
    model.compile(
      optimizer=optim,
        loss = 'mean_squared_error'
    )
    return(model,callbacks)