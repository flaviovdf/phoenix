#-*- coding: utf8
'''
This module contains the mixture of odes used for each
infectivity model.
'''
from __future__ import division, print_function

from numpy.linalg import LinAlgError

from phoenix import ode
from phoenix.peak_finder import find_peaks

import lmfit
import multiprocessing as mp
import numpy as np

RESIDUALS = set(['ssq', 'log', 'chisq'])
SCORES = set(['bic', 'msq'])

def period_fun(period, amp, phase, t):
    '''
    Simulates a sine wave to account for periodicity.

    Parameters
    ----------
    period : double
        the period of the sine wave

    amp : double
        the amplitude of the sine wave

    phase : double
        used to correct the period. For example, if the data starts on a 
        monday and has a seven day period, we need the phase to sync the
        sine wave to peak on wednesdays.

    t : array like or number
        the time tick to evaluate the period function
    '''
    
    return 1 - .5 * amp * (np.sin(2 * np.pi * (t + phase) / period) + 1)

def phoenix_r_with_period(parameters, num_ticks):
    '''
    Calls the Phoenix-R model adding a period function
    
    Parameters
    ----------
    parameters : dict like
        the parameters for the phoenix-r equations. See `ode.phoenix_r` for
        details.

    num_ticks : int
        number of ticks to simulate

    See Also
    --------
    ode.phoenix_r : for the actual phoenix R equations
    '''

    result = ode.phoenix_r(parameters, num_ticks)
    
    if isinstance(parameters, lmfit.Parameters):
        amp = parameters['amp'].value
        phase = parameters['phase'].value
        period = parameters['period'].value
    else:
        amp = parameters['amp']
        phase = parameters['phase']
        period = parameters['period']
        
    result *= period_fun(period, amp, phase, np.arange(num_ticks))

    return result

def residual(params, tseries, err_metric='log'):
    '''
    Computes the residual of the model. Different strategies can me used
    for this computaion such as:

        1. ssq (Sum Squared Errors) - Returns the sum of squared errors of the 
            model
        2. log (Sum Squared Errors on log) - The same as ssq but transforms the
            data and model to log scales. Useful for when the data has a high
            variability in values. This is close to minimizing the squared of
            the relative error
        3. chisq (Chi Squared) - Minimizes Pearson's Chi Square statistic. This
            is: $(tseries[i] - model[i])^2 / data[i]$. Also mimicks the relative
            error.
    
    Parameters
    ----------
    params : dict like
        Parameters to input to the model

    tseries : array like 
        Time series which the model tries to capture
    
    err_metric : string in ('ssq', 'log', 'chisq')
        Error metric to use, defaults to ssq

    Returns
    -------
    This method returns an array with:
        1. $tseries[i] - model[i]$ in the case of 'ssq'.
        2. $log(tseries[i]) - log(model[i]$ in the case of 'log'. 
            Zeros are masked out.
        3. $(tseries[i] - model[i])^2 / data[i]$ in the case 
            of 'chisq'. Zeros are masked out.

    To compute the actual error (ssq, log or chisq) one can do simply
    sum the squares of the return value. We return individual diferences
    since it is required by scipy's optimization algorithms.
    '''
    
    if err_metric not in RESIDUALS:
        raise ValueError('Most choose residual from ' + RESIDUALS)
    
    est = phoenix_r_with_period(params, tseries.shape[0])
    
    if err_metric in ('log', 'chisq'):
        msk = (est > 0) & (tseries > 0)
        if not msk.any():
            return (tseries - tseries.mean())
        
        if err_metric == 'log':
            return np.log(tseries[msk]) - np.log(est[msk])
        else:
            return tseries[msk] - est[msk] / np.sqrt(tseries[msk])
    else:
        return tseries - est

def msq(y_predicted, y_true, num_parameters):

    return ((y_true - y_predicted) ** 2).mean()

def bic(y_predicted, y_true, num_parameters):

    n = y_true.shape[0]
    k = num_parameters

    return n * np.log(((y_true - y_predicted) ** 2).mean()) + k * np.log(n)

def _params_to_list_of_tuples(params, ignore=None):
    
    copy = []
    for key in params:
        parameter = params[key]
        if ignore and parameter.name in ignore:
            continue

        copy.append((\
                    parameter.name, parameter.value, parameter.vary, \
                    parameter.min, parameter.max, parameter.expr))
    
    return copy

def _fit_one(tseries, period, err_metric, curr_sp=0, curr_params=None):

    init_params = []
    #copy current parameters
    if curr_params is not None:
        ignore = ('num_models', 'start_points')
        init_params.extend(_params_to_list_of_tuples(curr_params, ignore))
    else:
        #On the first run we search for a period
        init_params.append(\
                ('period', period, False, None, None, None))
        init_params.append(\
                ('amp', np.random.rand(), True, 0, 2, None))
        init_params.append(\
                ('phase', np.random.rand(), True, 0, period, None))
    
    #Add the new shock
    init_params.append(\
            ('i0_%d' % curr_sp, 1, False, None, None))
    init_params.append(\
            ('sp_%d' % curr_sp, curr_sp, False, None, None))
    init_params.append(\
            ('gamma_%d' % curr_sp, np.random.rand(), True, 0, 1))
    init_params.append(\
            ('beta_%d' % curr_sp, np.random.rand(), True, 0, 1))
    init_params.append(\
            ('r_%d' % curr_sp, np.random.rand(), True, 0, None))

    #Add the num models and start points params
    if curr_params and 'start_points' in curr_params:
        start_points = curr_params['start_points'].value
    else:
        start_points = []

    start_points.append(curr_sp)
    num_models = len(start_points)
    init_params.append(('start_points', start_points, False, None, None, None))
    init_params.append(('num_models', num_models, False, None, None, None))

    #Grid search for s0
    best_err = np.inf
    best_params = None
    for s_0 in np.logspace(1, 8, 15):
        params = lmfit.Parameters()
        params.add_many(*init_params)
        params.add('s0_%d' % curr_sp, value=s_0, vary=False)
        
        try:
            lmfit.minimize(residual, params, \
                    args=(tseries, err_metric))
            
            err = (residual(params, tseries, err_metric) ** 2).sum()
            if err < best_err:
                best_err = err
                best_params = params

        except (LinAlgError, FloatingPointError, ZeroDivisionError):
            continue
    
    #ugly ugly hack. stick with last guess if none worked
    if best_params is None:
        best_params = params
    
    return best_params

class FixedParamsPhoenixR(object):
    '''
    PhoenixR model with parameters

    Parameters
    ----------
    parameters : dict like
        The parameters for the model

    score_func : string in {'bic', 'msq'}
        Select the score to store. BIC or SSQ. Our BIC score is based
        on the assumptions that errors are normally distributed. This is
        considered a safe choice.
    '''
    def __init__(self, parameters, score_func='bic'):
        
        if score not in SCORES:
            raise ValueError('Most choose residual from ' + SCORES)

        self.parameters = parameters
        if score_func == 'bic':
            self.score_func = bic
        else:
            self.score_func = msq

        self.num_params = None
        self.score = None

    def __call__(self, num_ticks):
        return phoenix_r_with_period(self.parameters, num_ticks)
    
    def fit(self, tseries):
        tseries = np.asanyarray(tseries)
        
        num_models = 0
        if isinstance(self.parameters, lmfit.Parameters):
            num_models = self.parameters['num_models'].value
        else:
            num_models = self.parameters['num_models']

        self.num_params = 5 * num_models + 2
        self.score = self.score_func(phoenix_r_with_period(self.parameters, \
                tseries.shape[0]), tseries, self.num_params)
        return self

class InitParametersPhoenixR(object):
    '''
    PhoenixR with given start values

    Parameters
    ----------
    parameters : dict like or lmfit.Parameters
        Initial parameters for the phoenix R model

    err_metric : string in ('log', 'ssq', 'chisq')
        Error metric to minimize on the residual. See the function
        residual for more details.


    score_func : string in {'bic', 'msq'}
        Select the score to store. BIC or SSQ. Our BIC score is based
        on the assumptions that errors are normally distributed. This is
        considered a safe choice.
    '''
    def __init__(self, parameters, err_metric='log', score_func='bic'):
        self.parameters = self._tolmfit(parameters)
        self.err_metric = err_metric
        
        if score_func == 'bic':
            self.score_func = bic
        else:
            self.score_func = msq
       
        self.num_params = None
        self.score = None

    def _tolmfit(self, parameters):
        if isinstance(parameters, lmfit.Parameters):
            return parameters

        #Make sure we have the vary and limits correct
        rv = lmfit.Parameters()
        for key in parameters:
            if 'gamma_' in key:
                rv.add(key, value=parameters[key], min=0, max=1)
            elif 'beta_' in key:
                rv.add(key, value=parameters[key], min=0, max=1)
            elif 'r_' in key:
                rv.add(key, value=parameters[key], min=0)
            elif 'amp' == key:
                rv.add(key, value=parameters[key], min=0)
            elif 'phase' == key:
                rv.add(key, value=parameters[key], min=0, \
                        max=parameters['period'])
            else:
                #The rest we don't vary
                rv.add(key, value=parameters[key], vary=False)

        return rv

    def __call__(self, num_ticks):
        return phoenix_r_with_period(self.parameters, num_ticks)

    def fit(self, tseries):
        
        tseries = np.asanyarray(tseries)
        
        old_state = np.seterr(all='raise')
        lmfit.minimize(residual, self.parameters, \
                    args=(tseries, self.err_metric))
        model = phoenix_r_with_period(self.parameters, tseries.shape[0])

        num_models = self.parameters['num_models']
        self.num_params = 5 * num_models + 2
        self.score = self.score_func(model, tseries, self.num_params)

        np.seterr(**old_state)
        return self

class FixedStartPhoenixR(object):
    '''
    PhoenixR model with fixed start points. The model will fit one start
    point at a time, adding new ones in order. A final fit is performed 
    to using the results of the previous ones as start points.
    
    Parameters
    ----------
    start_points : array like
        List of start points for each infection. The algorithm will fit each
        start point in the order it appears on this list.

    period : integer
        Period to consider. If time windows are daily, 7 means weekly period

    err_metric : string in ('log', 'ssq', 'chisq')
        Error metric to minimize on the residual. See the function
        residual for more details.

    score_func : string in {'bic', 'msq'}
        Select the score to store. BIC or SSQ. Our BIC score is based
        on the assumptions that errors are normally distributed. This is
        considered a safe choice.
    '''

    def __init__(self, start_points, period=7, err_metric='log', \
            score_func='bic'):
        self.start_points = np.asanyarray(start_points, dtype='i')
        self.period = period
        self.err_metric = err_metric
        
        if score_func == 'bic':
            self.score_func = bic
        else:
            self.score_func = msq
        
        self.parameters = None
        self.num_params = None
        self.score = None
    
    def __call__(self, num_ticks):
        return phoenix_r_with_period(self.parameters, num_ticks)
    
    def fit(self, tseries):
        
        tseries = np.asanyarray(tseries)
        start_points = [sp for sp in self.start_points]

        old_state = np.seterr(all='raise')
        params = None
        for sp in start_points:
            params = _fit_one(tseries, self.period, self.err_metric, sp, params)
        
        num_models = len(start_points)
        self.num_params = 5 * num_models + 2
        self.parameters = params
        
        try:
            model = phoenix_r_with_period(self.parameters, tseries.shape[0])
        except (LinAlgError, FloatingPointError, ZeroDivisionError):
            model = tseries.mean()

        self.score = self.score_func(model, tseries, self.num_params)
        np.seterr(**old_state)
        return self

class WavePhoenixR(object):
    '''
    PhoenixR model with fixed start points
    
    Parameters
    ----------
    period : integer
        Period to consider. If time windows are daily, 7 means weekly period
    
    wave_widths : array like
        Widths to test while searching for peaks and start points

    threshold : double
        Will continue improving models while the error is decaying above
        this threshold.

    err_metric : string in ('log', 'ssq', 'chisq')
        Error metric to minimize on the residual. See the function
        residual for more details.
    
    score_func : string in {'bic', 'msq'}
        Select the score to store. BIC or SSQ. Our BIC score is based
        on the assumptions that errors are normally distributed. This is
        considered a safe choice. 
    '''

    def __init__(self, period=7, wave_widths=[1, 2, 4, 8, 16, 32, 64, 128, 256], 
            threshold=.05, err_metric='log', score_func='bic'):
        self.period = period
        self.wave_widths = wave_widths
        self.threshold = threshold
        self.err_metric = err_metric
        
        if score_func == 'bic':
            self.score_func = bic
        else:
            self.score_func = msq
        
        self.parameters = None
        self.num_params = None
        self.score = None

    def __call__(self, num_ticks):
        return phoenix_r_with_period(self.parameters, num_ticks)

    def fit(self, tseries):
        
        tseries = np.asanyarray(tseries)
        
        #i'm too lazy to type self a lot
        wave_widths = self.wave_widths 
        period = self.period
        threshold = self.threshold
        err_metric = self.err_metric
        score_func = self.score_func

        #Find peaks and add them all to a set 
        peaks = find_peaks(tseries, wave_widths)

        #Consider unique start_points only.
        #TODO: maybe we can use multiple infections starting at the same point?
        candidate_start_points = []
        candidate_start_points.append(0) 
        
        for x in peaks:
            candidate_sp = max(x[2] - x[1], 0)
            if candidate_sp not in candidate_start_points:
                candidate_start_points.append(candidate_sp)
        
        curr_score = np.finfo('d').max
        best_score = np.finfo('d').max
        best_params = None
        
        params = None
        for i in xrange(len(candidate_start_points)):
            sp = candidate_start_points[i]
            params = _fit_one(tseries, period, err_metric, sp, params)
            
            model = phoenix_r_with_period(params, tseries.shape[0])
            num_params = 5 * (i + 1) + 2
            curr_score = self.score_func(model, tseries, num_params)

            if (curr_score <= best_score):
                assert params is not best_params
                best_params = params
                best_score = curr_score
            else:
                increased_score = (curr_score - best_score) / best_score
                if increased_score > threshold:
                    break
        
        self.parameters = best_params
        self.num_params = 5 * self.parameters['num_models'].value + 2
        self.score = best_score

        return self
