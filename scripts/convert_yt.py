#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

from phoenix.basic_io import youtubeh5io 

import numpy as np
import os
import pandas as pd
import plac

DAY = 24 * 24 * 60

def main(out_folder):
   
    pops, ages, up_dates, good_series, good_videos = \
            youtubeh5io.get_good_videos()
    
    store = pd.HDFStore(os.path.join(out_folder, 'youtube-tseries.h5'))
    
    for i in xrange(len(pops)):

        pop_series = good_series[i] 
        up_date = up_dates[i]
        up_date /= 1000
        
        ticks = np.array([up_date + j * DAY for j in xrange(pop_series.shape[0])])
        days = pd.to_datetime(ticks, unit='s')
        ts_pop = pd.Series(pop_series, index=days)
        
        store['vid_%s' % good_videos[i]] = ts_pop

    store.close()

if __name__ == '__main__':
    plac.call(main)
