#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

from scipy import stats as ss

import numpy as np
import os
import pandas as pd
import plac

def print_results(results):
    
    n = len(results)
    print('mean', results.mean())
    print('median', np.median(results))
    print('25perc', np.percentile(results, 25) if n > 2 else np.nan)
    print('75perc', np.percentile(results, 75) if n > 2 else np.nan)
    print()

def main(input_fpath, out_folder):
    
    store = pd.HDFStore(input_fpath)
    results = []
    for obj_key in store.keys():
        h_frame = store[obj_key]
        h_time_series = h_frame.values
        
        pops = np.log(h_time_series[:, 0] + 1e-6)
        dups = np.log(h_time_series[:, 1] + 1e-6)
        audi = np.log(h_time_series[:, 2] + 1e-6)
        
        results.append(ss.pearsonr(audi.cumsum(), dups.cumsum())[0])
    store.close()
    
    print_results(np.asarray(results))

if __name__ == '__main__':
    plac.call(main)
