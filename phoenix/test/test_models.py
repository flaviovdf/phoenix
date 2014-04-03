# -*- coding: utf8
from __future__ import division, print_function

from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from phoenix import models
from phoenix import ode

import numpy as np

def tests_the_phoenix_r_method():
    '''Tests the phoenix r w period model'''
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
    
    parameters['period'] = 1
    parameters['amp'] = 0
    parameters['phase'] = 0

    results = models.phoenix_r_with_period(parameters, 200)
    
    assert_array_equal(exp_tseries, results)

def tests_the_phoenix_r_method_2():
    '''Tests the phoenix r w period model: forced period'''
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
    
    parameters['period'] = 1
    parameters['amp'] = 1
    parameters['phase'] = 2

    results = models.phoenix_r_with_period(parameters, 200)
    
    assert (results != exp_tseries).any()

def test_fixed_fit():
    '''Tests the fitting method for the fixed phoenix model'''

    exp_tseries = np.zeros(150, dtype='d') 
    result_1 = ode.shock(0.01, 0, 0.8, 1000, 2, 200)
    result_2 = ode.shock(0.02, 0, 0.8, 200, 3, 150)

    exp_tseries = result_1
    exp_tseries[50:] += result_2
   
    pv1, pv2 = sorted(exp_tseries[::-1])[:2]
    model = models.FixedStartPhoenixR([0, 50], [pv1, pv2], period=1)
    model.fit(exp_tseries)
    
    results = model(200)
    
    assert_array_equal([0, 50], model.parameters['start_points'].value)
    assert_equal(exp_tseries.shape, results.shape)

def test_init_params_model():
    
    exp_tseries = np.zeros(150, dtype='d') 
    result_1 = ode.shock(0.01, 0, 0.8, 1000, 2, 200)
    result_2 = ode.shock(0.02, 0, 0.8, 200, 3, 150)

    exp_tseries = result_1
    exp_tseries[50:] += result_2
   
    pv1, pv2 = sorted(exp_tseries[::-1])[:2]
    model = models.FixedStartPhoenixR([0, 50], [pv1, pv2], period=1)
    model.fit(exp_tseries)
    
    model2 = models.InitParametersPhoenixR(model.parameters)
    model2.fit(exp_tseries)

    results1 = model(200)
    results2 = model(200)
    
    assert_equal(exp_tseries.shape, results1.shape)
    assert_equal(exp_tseries.shape, results2.shape)
    
    for key in model.parameters:
        assert_equal(model.parameters[key].value, model2.parameters[key].value)

def test_wavelet_fit():
    '''Tests the wavelet fitting strategy'''

    exp_tseries = np.zeros(150, dtype='d') 
    result_1 = ode.shock(0.01, 0, 0.8, 1000, 2, 200)
    result_2 = ode.shock(0.02, 0, 0.8, 200, 3, 150)

    exp_tseries = result_1
    exp_tseries[50:] += result_2

    model = models.WavePhoenixR()
    model.fit(exp_tseries)

    results = model(200)
    
    assert 'beta_0' in model.parameters
    assert_equal(exp_tseries.shape, results.shape)
