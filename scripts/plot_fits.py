#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

from collections import OrderedDict

from matplotlib import pyplot as plt

from phoenix import models

import json
import numpy as np
import os
import pandas as pd
import plac
import sys 

def main(tseries_fpath, result_fpath, window_size='1d'):
   
    store = pd.HDFStore(tseries_fpath)
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
    
    #from random import shuffle
    keys = [x for x in parameters]
    #shuffle(keys)
    for key in keys:
        h_frame = store[key]
        d_frame = h_frame.resample(window_size, how='sum')
        
        if d_frame.values.ndim > 1:
            pop_series = d_frame[0]
            audi_series = d_frame[2]
        else:
            pop_series = d_frame
         
        params = parameters[key]
        model = models.FixedParamsPhoenixR(params)
        model.fit(pop_series.values) 
        pm = model(pop_series.shape[0])
        
        assert pm.shape == pop_series.shape
        
        pm_series = pd.Series(pm, index=pop_series.index)
        
        #pm_series = pm_series[pop_series.values > 0]
        #pop_series = pop_series[pop_series.values > 0]
        #new_series = new_series[pop_series.values > 0]

        pop_series.plot(logy=True, use_index=True, marker='o', color='w', label='data')
        pm_series.plot(logy=True, use_index=True, label='model', color='k')

        plt.legend()
        plt.show()

    store.close()
if __name__ == '__main__':
    plac.call(main)
