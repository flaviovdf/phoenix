# -*- coding: utf8
from __future__ import division, print_function

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from phoenix import ode

import numpy as np

def test_simulate_si():
    '''Tests the basic SI infectivity model'''
    
    beta = 0.01
    gamma = 0
    r = 1

    pop = 100
    s_0 = 98
    i_0 = 2

    result = ode.shock(beta, gamma, r, s_0, i_0, 100)
    
    assert_almost_equal(i_0 + s_0 * i_0 * beta, result[0])
    assert_almost_equal(100, result[-1])

def test_simulate_si_beta2():
    '''Tests the basic SI infectivity model with large beta'''
    
    beta = 0.02
    gamma = 0
    r = 1

    pop = 100
    s_0 = 98
    i_0 = 02

    result = ode.shock(beta, gamma, r, s_0, i_0, 100)
    
    assert_almost_equal(i_0 + s_0 * i_0 * beta, result[0])
    assert_almost_equal(100, result[-1])

def test_simulate_si_revisits():
    '''Tests the basic SIV infectivity model'''
    
    beta = 0.01
    gamma = 0.05
    r = 0.8

    pop = 100
    s_0 = 98
    i_0 = 02

    n = 200

    result_rev = ode.shock(beta, gamma, r, s_0, i_0, n)
    result_no_rev = ode.shock(beta, gamma, 0, s_0, i_0, n)
    result_all_rev = ode.shock(beta, gamma, 1, s_0, i_0, n)
    
    assert (result_rev >= result_no_rev).all()
    assert (result_rev <= result_all_rev).all()

def test_phoenix_model():
    '''Tests the multiple Phoenix-r model simulation'''
    
    exp_tseries = np.zeros(150, dtype='d')
    
    result_1 = ode.shock(0.01, 0, 0.8, 1000, 2, 200)
    result_2 = ode.shock(0.02, 0, 0.8, 200, 3, 150)

    exp_tseries = result_1
    exp_tseries[50:] += result_2
    
    parameters = {}
    parameters['beta_0'] = 0.01
    parameters['s0_0'] = 1000
    parameters['i0_0'] = 2
    parameters['sp_0'] = 0
    parameters['r_0'] = 0.8
    parameters['gamma_0'] = 0
    
    parameters['beta_50'] = 0.02
    parameters['s0_50'] = 200
    parameters['i0_50'] = 3
    parameters['sp_50'] = 50
    parameters['r_50'] = 0.8
    parameters['gamma_50'] = 0
    
    parameters['start_points'] = [0, 50]
    parameters['num_models'] = 2
 
    results = ode.phoenix_r(parameters, 200)
    assert_array_equal(exp_tseries, results)
