#!/usr/bin/env python
# -*- coding: utf8

from __future__ import division, print_function

from collections import defaultdict

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.spines import Spine
from matplotlib.ticker import MultipleLocator
from scipy import stats as ss

from vod.stats.curves import ecdf
from vod.stats.curves import epdf

import numpy as np
import os
import powerlaw
import tables
import plac

def initialize_matplotlib(): 
    inches_per_pt = 1.0 / 72.27 
    golden_mean = (np.sqrt(5) - 1.0) / 2.0 
    
    fig_width = 118.0 * inches_per_pt
    fig_height = .6 * fig_width

    rc('axes', labelsize=5) 
    rc('axes', titlesize=5) 
    rc('axes', unicode_minus=False) 
    rc('axes', grid=False) 
    rc('figure', figsize=(fig_width, fig_height)) 
    rc('grid', linestyle=':') 
    rc('font', family='serif') 
    rc('legend', fontsize=5) 
    rc('lines', linewidth=1) 
    rc('ps', usedistiller='xpdf') 
    rc('text', usetex=True) 
    rc('xtick', labelsize=5) 
    rc('ytick', labelsize=5)
    rc('xtick', direction='out') 
    rc('ytick', direction='out')

def get_hist(data):
    log_min_size = np.log10(data.min())
    log_max_size = np.log10(data.max())
    nbins = np.ceil((log_max_size - log_min_size) * 10)
    bins = np.unique(np.floor(np.logspace(log_min_size, log_max_size, nbins)))
    hist, edges = np.histogram(data, bins, density=True)
    bin_centers = (edges[1:] + edges[:-1]) / 2.0
   
    return bin_centers, hist

def four_plots(data, data_name, fname, logx=True):
    
    data = np.asanyarray(data) 
    data = data[data > 0]
    
    fit = powerlaw.Fit(data, xmin=[1, 100])
    
    #cdf_x, cdf_y = ecdf(data)
    #ccdf_y = 1 - cdf_y
    #odds_ratio = cdf_y[ccdf_y != 0] / ccdf_y[ccdf_y != 0]
    
    plt.xlabel(data_name, labelpad=0)
    plt.ylabel(r'$p(X = x)$', labelpad=0)
    ax = plt.gca()
    
    bin_centers, hist = get_hist(data)
    plt.loglog(bin_centers, hist, 'wo', ms=3, label='data')
    fit.power_law.plot_pdf(ax=plt.gca(), color='g', linestyle='-', label='plaw/lognrm')
    fit.lognormal.plot_pdf(ax=plt.gca(), color='b', linestyle='--')
    plt.legend(loc='lower left', frameon=False)

    plt.tight_layout(pad=0)
    plt.savefig(fname)
    plt.close()
    
    print(fname)
    for i in ['power_law', 'lognormal']:
        for j in ['power_law', 'lognormal']:
            if i != j:
                print(i, j)
                print(fit.distribution_compare(i, j))
                print()
                print()
    
    print('xmin', fit.xmin)
    
    d = fit.power_law
    print('Plaw - parameters D=', d.D)
    print('alpha', d.alpha)
    print()

    d = fit.lognormal
    print('Lognorm - parameters D=', d.D)
    print(d.parameter1_name, d.parameter1)
    print(d.parameter2_name, d.parameter2)
    print(d.parameter3_name, d.parameter3)
    print()

def main(input_fpath, table_name, out_folder):
   
    initialize_matplotlib()

    h5_file = tables.open_file(input_fpath)
    table = h5_file.get_node('/', table_name)
    
    obj_to_users = defaultdict(set)
    obj_count = defaultdict(int)
    duplicate_counts = defaultdict(int)
    audience_counts = defaultdict(int)

    for row in table:
        user = row['user_id']
        obj = row['object_id']
        
        if user in obj_to_users[obj]:
            duplicate_counts[obj] += 1
        else:
            audience_counts[obj] += 1
            
        obj_to_users[obj].add(user)
        obj_count[obj] += 1
    
    obj_counts_arr = np.zeros(len(obj_count), dtype='f')
    audience_counts_arr = np.zeros(len(obj_count), dtype='f')
    duplicate_counts_arr = np.zeros(len(obj_count), dtype='f')
    
    for obj in obj_count:
        obj_counts_arr[obj] = obj_count[obj]
        audience_counts_arr[obj] = audience_counts[obj]
        duplicate_counts_arr[obj] = duplicate_counts[obj]

    msk = obj_counts_arr > 500
    obj_counts_arr = obj_counts_arr[msk]
    audience_counts_arr = audience_counts_arr[msk]
    duplicate_counts_arr = duplicate_counts_arr[msk]

    dups_over_audi = duplicate_counts_arr / audience_counts_arr
    dups_over_pop = duplicate_counts_arr / obj_counts_arr
    
    
    print('Corr')
    print(ss.pearsonr(np.log10(audience_counts_arr[duplicate_counts_arr > 0]), \
        np.log10(duplicate_counts_arr[duplicate_counts_arr > 0])))
    print(ss.kendalltau(audience_counts_arr, duplicate_counts_arr))
    
    #fname = os.path.join(out_folder, '%s-rev_div_audi-xmin.pdf' % table_name)
    #four_plots(duplicate_counts_arr / audience_counts_arr, r'Revisits/Audience - $x$', fname, True)
    
    #print()
    #print('Fraction Dups > Audi', (dups_over_audi > 1).sum() / dups_over_audi.shape[0])
    #print('Median Dups / Audi', np.median(dups_over_audi))
    #print('Median Dups / Pop', np.median(dups_over_pop))
    #print()

    h5_file.close()

if __name__ == '__main__':
    plac.call(main)
