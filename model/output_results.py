'''
Output the results based on the model previously trained
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
from sklearn.utils import resample
from sksurv.metrics import concordance_index_ipcw 
from sksurv.metrics import cumulative_dynamic_auc
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

def output_bootstrap(model, n_iterations, df_train, data_train, y_train, df_test,name):
    """ Compute the output of the model on the bootstraped test set
    # Arguments
        model: neural network model trained with final parameters.
        n_iterations: number of bootstrap iterations
        df_train: training dataset
        data_train: two columns dataset with survival time and censoring status for training samples
        y_train: survival time
        df_test: test dataset
        name: name of the model
    # Returns
        results_all: AUC and Uno C-index at 5 and 10 years 
    """
    if name == "CoxTime" or name == "Cox-CC":
        _ = model.compute_baseline_hazards()
    results_all = pd.DataFrame(columns=['auc5', 'auc10', 'unoc5', 'unoc10'])
    results_final = pd.DataFrame(columns=['mean','ci95_lo','ci95_hi','std', 'count'])
    
    for i in range(n_iterations):
        print(i)
        test_boot = resample(df_test, n_samples=len(df_test), replace = True)
        x_test_boot = test_boot.drop(['surv_test','cen_test'], axis = 1)
        duration_test_b, event_test_b = test_boot['surv_test'].values, test_boot['cen_test'].values
        data_test_b = skSurv.from_arrays(event=event_test_b, time=duration_test_b)
        if name =="Cox-CC" or name == "CoxTime" or name == "DeepHit":
            surv = model.predict_surv_df(np.array(x_test_boot, dtype = 'float32'))
        else:
            n_picktime = int(y_train[['s']].apply(pd.Series.nunique))
            x_test_boot_all = pd.concat([x_test_boot]*n_picktime)
            time_test = pd.DataFrame(np.repeat(np.unique(y_train[['s']]),len(x_test_boot)))
            x_test_boot_all.reset_index(inplace=True, drop=True)
            x_test_boot_all = pd.concat([x_test_boot_all, time_test], axis = 1)
            surv = make_predictions_pseudobs(model, y_train, x_test_boot_all, x_test_boot,name)
        
        time_grid = np.linspace(duration_test_b.min(), duration_test_b.max(), 100)
        prob_5_10 = pd.concat([determine_surv_prob(surv,i) for i in (duration_test_b.min(),5,10)], axis = 1)
        auc5 = float(cumulative_dynamic_auc(data_train,data_test_b, -prob_5_10.iloc[:,1], 5)[0])
        auc10 = float(cumulative_dynamic_auc(data_train,data_test_b, -prob_5_10.iloc[:,2], 10)[0])
        unoc5 = float(concordance_index_ipcw(data_train,data_test_b, -prob_5_10.iloc[:,1], 5)[0])
        unoc10 = float(concordance_index_ipcw(data_train,data_test_b, -prob_5_10.iloc[:,2], 10)[0])
        results = pd.DataFrame({'auc5': [auc5], 'auc10': [auc10], 'unoc5': [unoc5], 'unoc10': [unoc10]})
        results_all = results_all.append(results, ignore_index=True, sort = False)

    for column in results_all :
        stats = results_all[column].agg(['mean', 'count', 'std'])
        scores = np.array(results_all[column])
        sorted_scores = np.sort(scores, axis=None)
        ci95_lo = sorted_scores[int(0.05 * len(sorted_scores))]
        ci95_hi = sorted_scores[int(0.95 * len(sorted_scores))]
        results_stat = pd.DataFrame({'mean':[stats[0]],  'ci95_lo' : ci95_lo,'ci95_hi' : [ci95_hi], 'std' : [stats[2]],'count':[stats[1]]})
        results_final = results_final.append(results_stat, ignore_index=False,sort=False)
    results_final.index = results_all.columns.tolist()
    return results_final

def make_predictions_pseudobs(model, y_train, x_test_all, x_test,name):
    """ Compute the predictions for neural network models with pseudo-observations
    # Arguments
        model: neural network model trained with final parameters.
        y_train: survival time
        x_test_all: input variables (one line for one patient at one timepoint)
        x_test: input variables (one line for one patient)
        name: name of the model
    # Returns
        surv: predictions for all the patients of the test set for the time points at which the pseudoobservations are computed
    """
    n_picktime = int(y_train[['s']].apply(pd.Series.nunique))
    y_pred = model.predict(x_test_all)
    y_pred = y_pred.reshape((n_picktime,len(x_test)))
    y_pred = pd.DataFrame(y_pred)

    if name == "pseudo-discrete":
        y_pred_all = pd.DataFrame()
        for j in range(len(y_pred.columns)):    
            for i in range(len(y_pred)):
                y_pred_all.loc[i,j] = y_pred.loc[:i,j].prod(axis = 0)
                surv = y_pred_all
    else:
        surv = y_pred
    surv = surv.set_index(np.unique(y_train[['s']]))
    return surv


def output_simulations(surv,df_train, x_test, df_test,name):
    """ Compute the output of the model on the test set
    # Arguments
        model: neural network model trained with final parameters.
        df_train: training dataset
        x_test: 20 simulated input variables
        df_test: test dataset
        name: name of the model
    # Returns
        results_test: AUC and Uno C-index at median survival time
    """
        
    data_train = skSurv.from_arrays(event=df_train['status'], time=df_train['yy'])
    data_test = skSurv.from_arrays(event=df_test['status'], time=df_test['yy'])
    cens_test = 100. - df_test['status'].sum() * 100. / df_test['status'].shape[0]
    
    time_med = np.percentile(data_test['time'], np.linspace(0, 50, 2))
    auc_med = float(cumulative_dynamic_auc(data_train, data_test, -determine_surv_prob(surv,time_med[1]), time_med[1])[0])
    unoc = float(concordance_index_ipcw(data_train,data_test, -determine_surv_prob(surv,time_med[1]), time_med[1])[0])
    
    results_test = pd.DataFrame({'t_med':time_med[1],
                                 'auc_med':[auc_med],
                                 'unoc' :[unoc],
                                 'cens_rate' : [cens_test]})
    return results_test





