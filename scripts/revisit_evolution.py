#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

from collections import defaultdict
from matplotlib import pyplot as plt

import numpy as np
import plac
import tables

def main(input_fpath, table_name):
    
    h5_file = tables.open_file(input_fpath)
    table = h5_file.get_node('/', table_name)
    
    obj_users = defaultdict(set)
    
    obj_revisits = []
    obj_new_visits = []
    
    first = True
    first_date = None
    last_date = None

    for row in table.itersorted('date'):
        user = row['user_id']
        obj = row['object_id']
        time_seconds = row['date']
        
        if first:
            first_date = time_seconds
            first = False
        last_date = time_seconds

        if user in obj_users[obj]:
            obj_revisits.append(1)
            obj_new_visits.append(0)
        else:
            obj_revisits.append(0)
            obj_new_visits.append(1)
            obj_users[obj].add(user)
    
    h5_file.close()

    sum_revists = np.cumsum(obj_revisits)
    sum_new_visits = np.cumsum(obj_new_visits)

    plt.plot(sum_revists / (sum_revists + sum_new_visits), 'b-', 
            label='rev/total')
    plt.plot(sum_revists / sum_new_visits, 'r-', label='rev/new')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plac.call(main)
