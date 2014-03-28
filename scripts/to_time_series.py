#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

from collections import defaultdict

import numpy as np
import pandas as pd
import os
import tables
import plac

DELTA = 60 * 60

def main(input_fpath, table_name, out_folder):
    h5_file = tables.open_file(input_fpath)
    table = h5_file.get_node('/', table_name)
    
    obj_to_users = defaultdict(set)
    obj_times = defaultdict(list)
    duplicate_times = defaultdict(list)
    audience_times = defaultdict(list)
    
    for row in table.itersorted('date'):
        user = row['user_id']
        obj = row['object_id']
        tstamp = row['date']
        
        if user in obj_to_users[obj]:
            duplicate_times[obj].append(tstamp)
        else:
            audience_times[obj].append(tstamp)
            
        obj_to_users[obj].add(user)
        obj_times[obj].append(tstamp)
    
    store = pd.HDFStore(os.path.join(out_folder, 
        '%s-tseries.h5' % table_name))
    for obj in range(len(obj_times)):
        if len(duplicate_times[obj]) < 2 or len(obj_times[obj]) < 500:
            continue
        
        occur_obj = np.ones(len(obj_times[obj]))
        occur_dups = np.ones(len(duplicate_times[obj]))
        occur_audi = np.ones(len(audience_times[obj]))
        
        min_date = pd.to_datetime(obj_times[obj][0], unit='s')
        max_date = pd.to_datetime(obj_times[obj][-1], unit='s')

        ts_pop = pd.Series(occur_obj, \
                        index=pd.to_datetime(obj_times[obj], unit='s'))
        
        ts_dups = pd.Series(occur_dups, \
                        index=pd.to_datetime(duplicate_times[obj], unit='s'))
        
        ts_audi = pd.Series(occur_audi, \
                        index=pd.to_datetime(audience_times[obj], unit='s'))
        
        ts_pop = ts_pop.resample('1h', how='sum').fillna(0)
        ts_dups = ts_dups.resample('1h', how='sum').fillna(0)
        ts_audi = ts_audi.resample('1h', how='sum').fillna(0)
        
        merged_series = pd.concat([ts_pop, ts_dups, ts_audi], \
                join='outer', axis=1).fillna(0)
        
        store['obj_%d' % obj] = merged_series
    store.close()

if __name__ == '__main__':
    plac.call(main)
