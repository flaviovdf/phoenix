#-*- coding: utf8
'''Score functions to evaluate models'''

from scipy.stats import norm

import numpy as np

DBL_COST = 8 * 8 #(64 bits)

def _fast_log2(x):
    x = int(np.ceil(x))
    return x.bit_length() - 1

def _fast_log2_star(x):
    if x <= 1:
        return 0

    #tail recursion should take care of large numbers
    return 1 + _fast_log2_star(_fast_log2(x)) 

def msq(y_predicted, y_true, num_parameters=None, parameters=None):

    return ((y_true - y_predicted) ** 2).mean()

def bic(y_predicted, y_true, num_parameters, parameters=None):
    
    n = y_true.shape[0]
    k = num_parameters

    return n * np.log(((y_true - y_predicted) ** 2).mean()) + k * np.log(n)

def mdl(y_predicted, y_true, num_parameters, parameters):
    
    n = y_true.shape[0]
    cost = 0
    
    start_points = parameters['start_points'].value
    num_start_points = len(parameters['start_points'].value)
    
    #cost for the data size
    cost += _fast_log2_star(n)

    #number of start_points (maximum of n)
    cost += _fast_log2_star(num_start_points)

    #each start point has a log*(n) cost
    cost += num_start_points * _fast_log2_star(n) 
    
    #period has log(7) cost
    cost += _fast_log2(parameters['period'].value)

    #Add the cost of each vary
    for param_name in parameters:
        p = parameters[param_name]
        if not p.vary:
            continue #we deal with constants separately

        upper_bound = p.max
        lower_bound = p.min
        value = p.value

        if 's0_' in param_name: #population size
            cost += _fast_log2_star(np.round(value))
        else: 
            #all other parameters are represented as doubles
            cost += DBL_COST
    
    diffs = y_true - y_predicted
    mean = diffs.mean()
    std = diffs.std()

    probs = norm.pdf(diffs, mean, std)
    surprisal = np.log2(1 / probs[probs != 0])
    cost += surprisal.sum()
    return cost
