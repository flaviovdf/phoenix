#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

from collections import OrderedDict

from matplotlib import pyplot as plt

import json
import numpy as np
import os
import plac
import sys 

def get_fits(input_fpath):
 
    parameters = OrderedDict()
    with open(input_fpath) as results_file:
        obj_id = None
        for i, line in enumerate(results_file):
            if i % 3 == 0:
                obj_id = line.strip()
            if i % 3 == 1:
                line = line.replace('array(', '')
                line = line.replace(', dtype=int32)', '')
                line = line.replace('\'', '"')
                params = json.loads(line.strip())
                parameters[obj_id] = params
            if i % 3 == 2:
                obj_id = None
    return parameters

def to_feature_mat(parameters):
    
    n_obj = len(parameters)
    X = np.zeros(shape=(n_obj, 8), dtype='f')

    for i, obj in enumerate(parameters):
        params = parameters[obj]

        models = set(params['start_points'])
        n_models = len(models)
        
        beta_key = 'beta_%d'
        gamma_key = 'gamma_%d'
        r_key = 'r_%d'
        s0_key = 's0_%d'
        
        amp = params['amp']
        shift = params['phase']
        
        avg_beta = 0
        avg_gamma = 0
        avg_r = 0
        avg_s0 = 0
        avg_R0 = 0
        avg_interest = 0
        for k in models:
            beta = params[beta_key % k]
            avg_beta += beta

            gamma = params[gamma_key % k]
            avg_gamma += gamma

            r = params[r_key % k]
            avg_r += r

            s0 = params[s0_key % k]
            avg_s0 += s0
            
            pop = params[s0_key % k] + 1

            if gamma > 0:
                r0 = pop * (beta / gamma)
            else:
                r0 = pop * beta
            avg_R0 += r0
            
            interest = r0 * r
            avg_interest += interest

        avg_beta /= n_models
        avg_gamma /= n_models
        avg_r /= n_models
        avg_s0 /= n_models
        avg_R0 /= n_models
        avg_interest /= n_models
        
        X[i] = [avg_beta, avg_gamma, avg_r, \
                avg_s0, avg_R0, avg_interest, amp, shift]

    return X

def main(input_fpath):
    
    parameters = get_fits(input_fpath)
    X = to_feature_mat(parameters)
    
    from sklearn.decomposition import RandomizedPCA
    #from sklearn.mixture import DPGMM
    from sklearn.preprocessing import StandardScaler

    #X = StandardScaler().fit_transform(X)
    
    #pca = RandomizedPCA(n_components=2)
    #pca.fit(X)
    #print(pca.explained_variance_ratio_)
    #T = pca.transform(X)
    #dpgmm = DPGMM(10, covariance_type='spherical', n_iter=200)
    
    #plt.plot(T[:, 0], T[:, 1], 'bo')
    #plt.show()
    #dpgmm.fit(X)
    #clusts = dpgmm.predict(X)
    #from collections import Counter
    #print(Counter(clusts))
    #print(set(clusts))
    plt.loglog(X[:, 1], X[:, 2], 'bo')
    #plt.hist(X[:, 1], log=True, bins=100)
    plt.show()

if __name__ == '__main__':
    plac.call(main)
