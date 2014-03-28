#!/usr/bin/env python
#-*- coding: utf8
from __future__ import division, print_function

from collections import defaultdict

from matplotlib import pyplot as plt

from vod.stats.curves import ecdf
from vod.stats.curves import epdf

import numpy as np
import tables
import plac

def four_plots(num_occur, fraction_repeat):
    
    num_occur = np.asanyarray(num_occur)
    fraction_repeat = np.asanyarray(fraction_repeat)
    
    cdf_x, cdf_y = ecdf(fraction_repeat)
    ccdf_y = 1 - cdf_y
    odds_ratio = cdf_y[ccdf_y != 0] / ccdf_y[ccdf_y != 0]

    plt.subplot(221)
    plt.plot(cdf_x, cdf_y)
    plt.xlabel('Fraction of Repeated')
    plt.ylabel('P(X < x)')
    
    plt.subplot(222)
    plt.plot(cdf_x, ccdf_y)
    plt.xlabel('Fraction of Repeated')
    plt.ylabel('P(X > x)')
    
    plt.subplot(223)
    plt.semilogy(cdf_x[ccdf_y != 0], odds_ratio)
    plt.xlabel('Fraction of Repeated')
    plt.ylabel('Odds Ratio: P(X < x) / P(X > x)')
    
    plt.subplot(224)
    plt.semilogx(num_occur, fraction_repeat, 'wo')
    plt.xlabel('# Occurrences')
    plt.ylabel('Fraction of Repeated')

    plt.tight_layout(pad=0)
    plt.show()
    plt.close()

def main(input_fpath, table_name):
    
    h5_file = tables.open_file(input_fpath)
    table = h5_file.get_node('/', table_name)
    
    obj_to_users = defaultdict(set)
    obj_count = defaultdict(int)
    duplicate_counts = defaultdict(int)

    for row in table:
        user = row['user_id']
        obj = row['object_id']
        
        if user in obj_to_users[obj]:
            duplicate_counts[obj] += 1
            
        obj_to_users[obj].add(user)
        obj_count[obj] += 1

    x = []
    y = []
    for obj in obj_count:
        if obj_count[obj] >= 100:
            x.append(obj_count[obj])
            y.append(duplicate_counts[obj] / obj_count[obj])
    
    four_plots(x, y)
    h5_file.close()

if __name__ == '__main__':
    plac.call(main)
