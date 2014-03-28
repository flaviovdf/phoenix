#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

from matplotlib import pyplot as plt

from phoenix import models

from pyseries.prediction.spikem import SpikeM

import numpy as np
import os
import plac
import sys 

def bic(x, y, k):
    n = x.shape[0]
    return n * np.log(((x - y) ** 2).mean()) + k * np.log(n)

rmsqerr = lambda x, y: np.sqrt(((x - y) ** 2).mean())

def load_data(fpath):
    D = []
    with open(fpath) as data_file:
        for i, line in enumerate(data_file):
            if i % 2 != 0:
                D.append([float(x) for x in line.split()])
    
    return np.asarray(D)

def main(input_fpath):
    
    D = load_data(input_fpath)
    for i in xrange(D.shape[0]):
        try:
            pop_series = D[i]

            model = models.MultiSIR()
            params = models.wavelet_fit(pop_series, period=24, widths=[1, 2, 6, 12, 24], 
                    max_models=4)
            phoenix_model = model(params)
            
            sm = SpikeM(steps_ahead=0).fit_predict(
                    [pop_series], full_series=True, period_frequencies=[7])[0]
            
            print(i)
            print(params)
            print(rmsqerr(pop_series, phoenix_model), \
                    bic(pop_series, phoenix_model, model.num_params), end=' ')
            print(rmsqerr(pop_series, sm), \
                    bic(pop_series, sm, 7), end=' ')
            print()
        except Exception as e:
            print('err at key', i, file=sys.stderr)

    store.close()
if __name__ == '__main__':
    plac.call(main)
