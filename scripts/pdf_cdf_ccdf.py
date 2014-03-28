#!/usr/bin/env python
# -*- coding: utf8

from __future__ import division, print_function

from collections import defaultdict

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.spines import Spine
from matplotlib.ticker import MultipleLocator

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
    
    fig_width = 2 * 240.0 * inches_per_pt
    fig_height = .25 * fig_width

    rc('axes', labelsize=8) 
    rc('axes', titlesize=7) 
    rc('axes', unicode_minus=False) 
    rc('axes', grid=False) 
    rc('figure', figsize=(fig_width, fig_height)) 
    rc('grid', linestyle=':') 
    rc('font', family='serif') 
    rc('legend', fontsize=8) 
    rc('lines', linewidth=1) 
    rc('ps', usedistiller='xpdf') 
    rc('text', usetex=True) 
    rc('xtick', labelsize=7) 
    rc('ytick', labelsize=7)
    rc('xtick', direction='out') 
    rc('ytick', direction='out')

def four_plots(data, data_name, fname):
    
    data = np.asanyarray(data) 
    fit = powerlaw.Fit(data, discrete=True, xmin=[1, 100])
    xmin = fit.xmin
    
    data_cut = data[data >= xmin]
    cdf_x, cdf_y = ecdf(data_cut)
    ccdf_y = 1 - cdf_y
    odds_ratio = cdf_y[ccdf_y != 0] / ccdf_y[ccdf_y != 0]

    log_min_size = np.log10(data_cut.min())
    log_max_size = np.log10(data_cut.max())
    nbins = np.ceil((log_max_size - log_min_size) * 10)
    bins = np.unique(np.floor(np.logspace(log_min_size, log_max_size, nbins)))
    hist, edges = np.histogram(data_cut, bins, density=True)
    bin_centers = (edges[1:] + edges[:-1]) / 2.0
    
    plt.subplot(131)
    plt.xlabel(data_name, labelpad=0)
    plt.ylabel(r'$p(X = x)$', labelpad=0)
    
    plt.loglog(bin_centers, hist, 'wo', ms=5)
    fit.power_law.plot_pdf(ax=plt.gca(), color='g', linestyle='-')
    fit.lognormal.plot_pdf(ax=plt.gca(), color='b', linestyle='--')
    fit.truncated_power_law.plot_pdf(ax=plt.gca(), color='r', linestyle=':')
    
    plt.subplot(132)
    plt.xlabel(data_name, labelpad=0)
    plt.ylabel(r'$P(X > x)$', labelpad=0)
    
    plt.loglog(cdf_x, ccdf_y, 'wo', ms=5, label='data', markevery=10)
    fit.power_law.plot_ccdf(ax=plt.gca(), color='g', linestyle='-.', label='powerlaw')
    fit.lognormal.plot_ccdf(ax=plt.gca(), color='b', linestyle='--', label='lognormal')
    fit.truncated_power_law.plot_ccdf(ax=plt.gca(), color='r', linestyle=':', label='powerlaw+cutoff')
    plt.legend(loc='lower left', frameon=False)
    
    plt.subplot(133)
    plt.xlabel(data_name, labelpad=0)
    plt.ylabel(r'Odds Ratio', labelpad=0)
    
    xvals = cdf_x[ccdf_y != 0]
    odds_plaw = fit.power_law.cdf(xvals) / fit.power_law.ccdf(xvals) 
    odds_lognorm = fit.lognormal.cdf(xvals) / fit.lognormal.ccdf(xvals) 
    odds_trunc = fit.truncated_power_law.cdf(xvals) / fit.truncated_power_law.ccdf(xvals) 

    plt.loglog(xvals, odds_ratio, 'wo', ms=5, markevery=10)
    plt.loglog(xvals, odds_plaw, 'g-')
    plt.loglog(xvals, odds_lognorm, 'b--')
    plt.loglog(xvals, odds_trunc, 'r:')
    
    plt.tight_layout(pad=0)
    plt.savefig(fname)
    plt.close()
    
    print(fname)
    for i in ['power_law', 'lognormal', 'truncated_power_law']:
        for j in ['power_law', 'lognormal', 'truncated_power_law']:
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

    d = fit.truncated_power_law
    print('ExpTrunc - parameters D=', d.D)
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
    
    fname = os.path.join(out_folder, '%s-pop-xmin.pdf' % table_name)
    four_plots(obj_count.values(), r'Popularity - $x$', fname)
    
    fname = os.path.join(out_folder, '%s-audi-xmin.pdf' % table_name)
    four_plots(audience_counts.values(), r'Audience - $x$', fname)
    
    fname = os.path.join(out_folder, '%s-rev-xmin.pdf' % table_name)
    four_plots(duplicate_counts.values(), r'Revisits - $x$', fname)
    
    h5_file.close()

if __name__ == '__main__':
    plac.call(main)
