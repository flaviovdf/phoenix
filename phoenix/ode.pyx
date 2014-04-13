#-*- coding: utf8
# cython:boundscheck=False
# cython:wraparound=False
'''
Cython module containing the Euler method for simulating the ODEs of the
Shock and Phoenix-R model
'''
from __future__ import division, print_function

from cython cimport parallel

import lmfit

import numpy as np
cimport numpy as np

cdef extern from 'math.h':
    double exp(double x) nogil

cdef double[:] _shock(double beta, double gamma, double alpha, double r, \
        long s_0, long i_0, double d_t, double[:] store_at, \
        Py_ssize_t start, Py_ssize_t end, int accumulate, \
        int store_audience) nogil:
    '''
    This method implements the euler method for simulating the shock model.

    Parameters
    ----------
    beta : float
        strength of the infection

    gamma : float
        strength of the recovery

    alpha : float
        strength of re-infection

    r : float
        revisit rate

    s_0 : long
        initial number of suscepted

    i_0 : long
        initial number of infected

    d_t : double
        size of eack tick. Lower values == more precision

    store_at : double array
        array to store results at

    start : int (Py_ssize_t)
        where to begin storing results
    
    start : int (Py_ssize_t)
        where to end storing results

    accumulate : 1 or 0
        set to 1 to add to the store_at array. 0 means replace.

    store_audience : 1 or 0
        indicates if we should store audience and not popularity
    '''
    
    cdef double pop_size = s_0 + i_0
    
    cdef double newly_infected = 0
    cdef double newly_recovered = 0
    cdef double newly_re_susceptible = 0
    
    cdef double susceptible = s_0
    cdef double infected = i_0
    cdef double recovered = 0

    cdef double d_s = 0
    cdef double d_r = 0
    cdef double d_i = 0
    
    cdef Py_ssize_t n_steps = end - start
    cdef Py_ssize_t t = 0
    cdef double rv_i = 0
    for t in range(n_steps):
        
        newly_infected = beta * (susceptible * infected) * d_t
        if newly_infected > susceptible:
            newly_infected = susceptible
        
        newly_re_susceptible = alpha * recovered * d_t
        if newly_re_susceptible > recovered:
            newly_re_susceptible = recovered

        newly_recovered = gamma * infected * d_t
        if newly_recovered > infected:
            newly_recovered = infected

        d_s = -newly_infected + newly_re_susceptible
        d_i = newly_infected - newly_recovered
        d_r = newly_recovered

        susceptible = susceptible + d_s
        if susceptible < 0:
            susceptible = 0
            
        infected = infected + d_i
        if infected < 0:
            infected = 0
        
        recovered = recovered + d_r
        if recovered < 0:
            recovered = 0
        
        if store_audience == 1:
            if gamma > 0:
                rv_i = (1 - exp(-r/gamma)) * d_r
            else:
                rv_i = d_r
        else:
            rv_i = infected * r
        
        if accumulate == 1:
            store_at[t + start] += rv_i
        else:
            store_at[t + start] = rv_i

    return store_at

cdef double[:] _phoenix_r(double[:, ::1] param_mat, double[:] store_at, \
        int store_audience) nogil:
    '''
    This method implemets the phoenix-r equations. Parameters are passed
    as a matrix, being each row shock parameters for individual shocks.

    Parameters
    ----------
    param_mat : num_shocks by 9 matrix.
        Each column of the matrix has the form:
            0. beta
            1. gamma
            2. alpha
            3. r
            4. s_0
            5. i_0
            6. d_t
            7. sp (start point of the shock)
            8. num_ticks

    store_at : array
        Where to store results
    
    store_audience : 1 or 0
        indicates if we should store audience and not popularity
    '''

    cdef Py_ssize_t num_models = param_mat.shape[0]
    cdef Py_ssize_t i
    for i in range(num_models):
        _shock(param_mat[i, 0], param_mat[i, 1], param_mat[i, 2],
                param_mat[i, 3], <long> param_mat[i, 4], <long> param_mat[i, 5],
                param_mat[i, 6], store_at, <Py_ssize_t> param_mat[i, 7],
                <Py_ssize_t> param_mat[i, 8], 1, store_audience)
    return store_at

def shock(double beta, double gamma, double r, long s_0, long i_0, \
        Py_ssize_t num_ticks, store_audience=False):
    '''
    This method implements a single shock simulation.
    This is a wrapper method for the faster cython code.

    Parameters
    ----------
    beta : float
        strength of the infection

    gamma : float
        strength of the recovery

    r : float
        revisit rate

    s_0 : long
        initial number of suscepted

    i_0 : long
        initial number of infected

    num_ticks : int
        number of ticks to simulate

    store_audience : bool (default=False)
        indicates if the model should return the audience
    '''
 
    store_audiece = int(store_audience)
    rv = np.zeros(num_ticks, dtype='d')
    d_t = 1.0 #we assume discrete time.
    alpha = 0 #no re-infections
    _shock(beta, gamma, alpha, r, s_0, i_0, d_t, rv, 0, num_ticks, 0, \
            store_audiece)
    return rv

def _unpack_params(parameters):
    '''Auxiliary method to convert lmfits parameters to a dict'''

    params_dict = {}

    if isinstance(parameters, lmfit.Parameters):
        for key, param in parameters.items():
            params_dict[key] = param.value
    else:
        params_dict = parameters

    return params_dict

def phoenix_r(parameters, num_ticks, store_audience=False):
    '''
    This method implements a single shock simulation.
    This is a wrapper method for the faster cython code.

    Parameters
    ----------
    parameters : dict like
        parameters for each individual shock
    
    num_ticks : int
        number of ticks to simulate
    
    store_audience : bool (default=False)
        indicates if the model should return the audience
    '''
    
    cdef dict params_dict = _unpack_params(parameters)

    num_models = params_dict['num_models']
    result = np.zeros(num_ticks, dtype='d')
    start_points = params_dict['start_points']

    cdef double[:, ::1] param_mat = np.zeros((num_models, 9), dtype='d')
    cdef Py_ssize_t i
    for i in range(num_models):
        model_num = start_points[i]
        param_mat[i, 0] = params_dict['beta_%d' % model_num]
        param_mat[i, 1] = params_dict['gamma_%d' % model_num]
        param_mat[i, 2] = 0 #no re-infections
        param_mat[i, 3] = params_dict['r_%d' % model_num]
        param_mat[i, 4] = params_dict['s0_%d' % model_num]
        param_mat[i, 5] = params_dict['i0_%d' % model_num]
        param_mat[i, 6] = 1 #dt = 1
        param_mat[i, 7] = params_dict['sp_%d' % model_num]
        param_mat[i, 8] = num_ticks
    
    rv = np.zeros(num_ticks, dtype='d')
    _phoenix_r(param_mat, rv, int(store_audience))
    return rv
