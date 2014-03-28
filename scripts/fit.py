#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

from matplotlib import pyplot as plt

from phoenix import models

from pyseries.prediction.state_space import SSM

import numpy as np
import os
import pandas as pd
import plac
import tables
import sys 

def bic(x, y, k):
    n = x.shape[0]
    return n * np.log(((x - y) ** 2).mean()) + k * np.log(n)

rmsqerr = lambda x, y: np.sqrt(((x - y) ** 2).mean())

def main(input_fpath, window_size='1d'):
   
    store = pd.HDFStore(input_fpath)
    with open(input_fpath + '.groups') as keys_file:
        keys = [line.split()[0] for line in keys_file]
    
    popularity = np.zeros(len(keys), dtype='f')
    for i, obj_key in enumerate(keys):
        
        h_frame = store[obj_key]
        d_frame = h_frame.resample(window_size, how='sum')
        
        if d_frame.values.ndim > 1:
            pop_series = d_frame.values[:, 0]
        else:
            pop_series = d_frame.values
        
        
        if pop_series.shape[0] < 30 or sum(pop_series > 0) < 10 or \
                sum(pop_series) < 10000:
            continue

        try:
            model = models.MultiSIR()
            params = models.wavelet_fit(pop_series, max_models=4)
            phoenix_model = model(params)
        
            ssm1 = SSM(True, True, steps_ahead=0)
            ssm1_model = ssm1.fit_predict(pop_series[None], True)[0]

            ssm2 = SSM(True, False, steps_ahead=0)
            ssm2_model = ssm2.fit_predict(pop_series[None], True)[0]
        
            ssm3 = SSM(False, True, steps_ahead=0)
            ssm3_model = ssm3.fit_predict(pop_series[None], True)[0]
        
            ssm4 = SSM(False, False, steps_ahead=0)
            ssm4_model = ssm4.fit_predict(pop_series[None], True)[0]
        
            print(obj_key)
            print(params)
            print(rmsqerr(pop_series, phoenix_model), \
                    bic(pop_series, phoenix_model, model.num_params), end=' ')
            print(rmsqerr(pop_series, ssm1_model), \
                    bic(pop_series, ssm1_model, 7), end=' ')
            print(rmsqerr(pop_series, ssm2_model), \
                    bic(pop_series, ssm2_model, 5), end=' ')
            print(rmsqerr(pop_series, ssm3_model), 
                    bic(pop_series, ssm3_model, 5), end=' ')
            print(rmsqerr(pop_series, ssm4_model), 
                    bic(pop_series, ssm4_model, 3), end=' ')
            print()
        except Exception as e:
            print('err at key', obj_key, file=sys.stderr)

    store.close()
if __name__ == '__main__':
    plac.call(main)
