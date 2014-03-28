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

def load_data(fpath):
    D = []
    with open(fpath) as data_file:
        for i, line in enumerate(data_file):
            if i % 2 != 0:
                D.append([float(x) for x in line.split()])
    
    return np.asarray(D)

def main(tseries_fpath, result_fpath):
   
    D = load_data(tseries_fpath)

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
    
    for i in parameters:
        params = parameters[i]
        model = models.MultiSIR()
        pm = model(params)
        
        pop_series = pd.Series(D[int(i)])
        pm_series = pd.Series(pm)

        assert pm_series.shape == pop_series.shape
        
        pop_series.plot(logy=True, use_index=True, marker='o', color='w', label='data')
        pm_series.plot(logy=True, use_index=True, label='model', color='k')
        
        plt.legend()
        plt.show()

    store.close()
if __name__ == '__main__':
    plac.call(main)
