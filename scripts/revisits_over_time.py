#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

import numpy as np
import os
import pandas as pd
import plac
import tables

def get_above(time_series):

    pops = time_series[:, 0]
    dups = time_series[:, 1]
    audi = time_series[:, 2]
    
    dups = dups[pops >= 20]
    audi = audi[pops >= 20]

    return dups / audi

def print_results(results_array):
    
    n = results_array.shape[0]
    msk_inf = results_array != np.inf
    num_infs = msk_inf.sum()

    no_infs = results_array[results_array != np.inf]

    print('Num time windows', n)
    print('% of time windows with 0 audience', 1 - (num_infs / n))
    print('mean of dups/audience', no_infs.mean())
    print('median of dups/audience', np.median(no_infs))
    print('25perc of dups/audience', np.percentile(no_infs, 25) if n > 2 else np.nan)
    print('75perc of dups/audience', np.percentile(no_infs, 75) if n > 2 else np.nan)
    print('std of dups/audience', no_infs.std())
    print()

def main(input_fpath, out_folder):
    
    store = pd.HDFStore(input_fpath)
    
    h_results = []
    d_results = []
    w_results = []
    m_results = []

    for obj_key in store.keys():
        
        h_frame = store[obj_key]
        d_frame = h_frame.resample('1d', how='sum')
        w_frame = d_frame.resample('1W', how='sum')
        m_frame = w_frame.resample('1M', how='sum')

        h_time_series = h_frame.values
        d_time_series = d_frame.values
        w_time_series = w_frame.values
        m_time_series = m_frame.values

        h_results.extend(get_above(h_time_series))
        d_results.extend(get_above(d_time_series))
        w_results.extend(get_above(w_time_series))
        m_results.extend(get_above(m_time_series))
    store.close()
    
    #infs mean that no unique visitors appeared in that time window
    h_results = np.asarray(h_results)
    d_results = np.asarray(d_results)
    w_results = np.asarray(w_results)
    m_results = np.asarray(m_results)

    print('Hourly')
    print_results(h_results)
   
    print('Daily')
    print_results(d_results)
    
    print('Weekly')
    print_results(w_results)
    
    print('Monthly')
    print_results(m_results)

if __name__ == '__main__':
    plac.call(main)
