#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

from collections import OrderedDict
from scipy import stats as ss

import json
import numpy as np
import os
import plac
import sys 

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri #Automagic conversion of numpy to R
rpy2.robjects.numpy2ri.activate()

def main(result_fpath):
   
    parameters = OrderedDict()
    errors = OrderedDict()
    with open(result_fpath) as results_file:
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
                err = np.asarray([float(x) for x in line.split()])
                errors[obj_id] = err
                obj_id = None
    
    bics_phx = []
    bics_kir = []
    wins = []
    diff = []
    for key in parameters:
        params = parameters[key]
        err = errors[key]

        #Fixing our BIC estimate, due to a bug in the code
        num_models_orig = len(params['start_points'])
        num_models_fix = len(set(params['start_points']))
        
        k = 2 + num_models_fix * 5
        n = params['num_ticks']
        
        bic_phoenix = err[1]
        bic_fix = bic_phoenix - (num_models_orig * np.log(n))
        bic_fix += (num_models_fix * np.log(n))
        
        bic_kir = min(err[[3, 5, 7, 9]])
        
        bics_phx.append(bic_fix)
        bics_kir.append(bic_kir)
        
        diff.append((bic_kir - bic_fix) / bic_kir)
        wins.append(bic_kir - bic_fix > 0)

    bics_phx = np.asarray(bics_phx)
    bics_kir = np.asarray(bics_kir)

    ks = robjects.r['ks.test']
    res = ks(bics_phx, bics_kir, alternative='less')
    val = res.rx2('statistic')[0]
    p_val = res.rx2('p.value')[0]
    
    print(val, p_val, np.median(diff), sum(wins), bics_phx.shape[0], sum(wins) / bics_phx.shape[0])
if __name__ == '__main__':
    plac.call(main)
