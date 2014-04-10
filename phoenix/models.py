#-*- coding: utf8
'''
This module contains the mixture of odes used for each
infectivity model. The classes here defined are used for
fitting the Phoenix-R model.
'''
from __future__ import division, print_function

from numpy.linalg import LinAlgError

from phoenix import ode

from phoenix.peak_finder import find_peaks

from phoenix.score_funcs import bic
from phoenix.score_funcs import msq
from phoenix.score_funcs import mdl 

import lmfit
import multiprocessing as mp
import numpy as np

RESIDUALS = set(['lin', 'log', 'mean'])
SCORES = {'bic':bic,
          'msq':msq,
          'mdl':mdl}

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

def residual(params, tseries, residual_metric):
    '''
    Computes the residual of the model. Different strategies can me used
    for this computaion such as:

        1. lin (Sum squared errors) - Returns the sum of squared errors of the 
            model
        2. log (Sum squared errors on log) - The same as msq but transforms 
            the data and model to log scales. Useful for when the data has a high
            variability in values. This is close to minimizing the squared of
            the relative error
        3. mean (mean squared errors) - Returns the mean squared errors of 
            the model
    
    Parameters
    ----------
    params : dict like
        Parameters to input to the model

    tseries : array like 
        Time series which the model tries to capture
    
    residual_metric : string in ('lin', 'log', 'mean')
        Error metric to use, defaults to mean

    Returns
    -------
    This method returns an array with:
        1. sum($tseries[i] - model[i]$) in the case of 'lin'.
        2. sum($log(tseries[i]) - log(model[i]$) in the case of 'log'.
        3. mean($tseries[i] - model[i]$) in the case of 'mean'.
    '''
    
    if residual_metric not in RESIDUALS:
        raise ValueError('Most choose residual from ' + ' '.join(RESIDUALS))

    est = phoenix_r_with_period(params, tseries.shape[0])

    if residual_metric == 'log':
        msk = (est > 0) & (tseries > 0)
        if not msk.any():
            return (tseries - tseries.mean())
        return np.log(tseries[msk]) - np.log(est[msk]) 
    else:
        n = tseries.shape[0]
        if residual_metric == 'mean':
            div = np.sqrt(n)
        else:
            div = 1
        return (tseries - est) / div

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

def _fit_one(tseries, period, residual_metric, curr_sp, curr_pv, curr_params):
    
    init_params = []
    #copy current parameters
    if curr_params is not None:
        ignore = ('num_models', 'start_points')
        init_params.extend(_params_to_list_of_tuples(curr_params, ignore))
    else:
        #On the first run we search for a period
        init_params.append(\
                ('period', period, False))
        init_params.append(\
                ('amp', 0, True, 0, 1))
        init_params.append(\
                ('phase', 0, 0, 0, period))
    
    #Add the new shock
    if curr_sp != 0:
        init_params.append(\
                ('s0_%d' % curr_sp, curr_pv, True))

    init_params.append(\
            ('i0_%d' % curr_sp, 1, False))
    init_params.append(\
            ('sp_%d' % curr_sp, curr_sp, False))
    init_params.append(\
            ('beta_%d' % curr_sp, np.random.rand(), True, 0, 1))
    init_params.append(\
            ('r_%d' % curr_sp, np.random.rand(), True, 0))
    init_params.append(\
            ('gamma_%d' % curr_sp, np.random.rand(), True, 0, 1))

    #Add the num models and start points params
    if curr_params and 'start_points' in curr_params:
        start_points = [x for x in curr_params['start_points'].value]
    else:
        start_points = []

    start_points.append(curr_sp)
    num_models = len(start_points)

    init_params.append(('start_points', start_points, False))
    init_params.append(('num_models', num_models, False))
    
    #Grid search for s0_0
    best_err = np.inf
    best_params = None
    for s0_0 in np.logspace(2, 6, 21):
        params = lmfit.Parameters()
        params.add_many(*init_params)
        params.add('s0_0', value=s0_0, vary=True)
        
        try:
            lmfit.minimize(residual, params, \
                    args=(tseries, residual_metric), ftol=.0001, xtol=.0001)
            err = (residual(params, tseries, residual_metric) ** 2).sum()
            if err < best_err:
                best_err = err
                best_params = params
    
        except (AssertionError, LinAlgError, FloatingPointError, \
                ZeroDivisionError):
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
        
        self.parameters = parameters
        self.score_func = SCORES[score_func]

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
                tseries.shape[0]), tseries, self.num_params, self.parameters)
        return self

class InitParametersPhoenixR(object):
    '''
    PhoenixR with given start values

    Parameters
    ----------
    parameters : dict like or lmfit.Parameters
        Initial parameters for the phoenix R model

    residual_metric : string in ('lin', 'log', 'mean')
        Error metric to minimize on the residual. See the function
        `residual` for more details.


    score_func : string in {'bic', 'msq'}
        Select the score to store. BIC or MSQ. Our BIC score is based
        on the assumptions that errors are normally distributed. This is
        considered a safe choice.
    '''
    def __init__(self, parameters, residual_metric='mean', score_func='bic'):
        self.parameters = self._tolmfit(parameters)
        self.residual_metric = residual_metric
        self.score_func = SCORES[score_func]
       
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
                    args=(tseries, self.residual_metric))
        model = phoenix_r_with_period(self.parameters, tseries.shape[0])

        num_models = self.parameters['num_models']
        self.num_params = 5 * num_models + 2
        self.score = self.score_func(model, tseries, self.num_params, \
                self.parameters)

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

    peak_volumes : array like
        The peak volumes for start point

    period : integer
        Period to consider. If time windows are daily, 7 means weekly period

    residual_metric : string in ('lin', 'log', 'mean')
        Error metric to minimize on the residual. See the function
        residual for more details.

    score_func : string in ('bic', 'msq')
        Select the score to store. BIC or SSQ. Our BIC score is based
        on the assumptions that errors are normally distributed. This is
        considered a safe choice.
    '''

    def __init__(self, start_points, peak_volumes, period=7, \
            residual_metric='mean', score_func='bic'):
        
        self.start_points = np.asanyarray(start_points, dtype='i')
        self.peak_volumes = np.asanyarray(peak_volumes, dtype='f')

        assert self.start_points.shape[0] == self.peak_volumes.shape[0]

        self.period = period
        self.residual_metric = residual_metric
        self.score_func = SCORES[score_func] 
        
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
        for i in xrange(self.start_points.shape[0]): 
            sp = self.start_points[i]
            pv = self.peak_volumes[i]
            
            params = \
                _fit_one(tseries, self.period, self.residual_metric, sp, pv, params)
        
        num_models = len(start_points)
        self.num_params = 5 * num_models + 2
        self.parameters = params
        
        try:
            model = phoenix_r_with_period(self.parameters, tseries.shape[0])
        except (LinAlgError, FloatingPointError, ZeroDivisionError):
            model = tseries.mean()

        self.score = self.score_func(model, tseries, self.num_params, \
                self.parameters)
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

    residual_metric : string in ('lin', 'log', 'mean')
        Error metric to minimize on the residual. See the function
        residual for more details.
    
    score_func : string in {'bic', 'msq'}
        Select the score to store. BIC or SSQ. Our BIC score is based
        on the assumptions that errors are normally distributed. This is
        considered a safe choice. 
    '''

    def __init__(self, period=7, wave_widths=[1, 2, 4, 8, 16, 32, 64, 128, 256], 
            threshold=.05, residual_metric='mean', score_func='bic'):
        
        self.period = period
        self.wave_widths = wave_widths
        self.threshold = threshold
        self.residual_metric = residual_metric
        self.score_func = SCORES[score_func]
        
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
        residual_metric = self.residual_metric
        score_func = self.score_func

        #Find peaks and add them all to a set 
        peaks = find_peaks(tseries, wave_widths)

        #Consider unique start_points only.
        #TODO: maybe we can use multiple infections starting at the same point?
        candidate_start_points = []
        candidate_start_points.append(0)
        
        min_sp = 1
        for x in peaks:
            min_sp = min(min_sp, max(x[2] - x[1], 1))

        #first peak is searched for by grid search 
        candidate_peak_volumes = []
        candidate_peak_volumes.append(1)
        
        for x in peaks:
            candidate_sp = max(x[2] - x[1], 1)
            peak_vol = max(tseries[x[2]] - tseries[candidate_sp], 0)
            
            if peak_vol == 0:
                continue

            if candidate_sp not in candidate_start_points:
                candidate_start_points.append(candidate_sp)
                candidate_peak_volumes.append(peak_vol)
        
        curr_score = np.finfo('d').max
        best_score = np.finfo('d').max
        best_params = None
        
        params = None
        for i in xrange(len(candidate_start_points)):
            sp = candidate_start_points[i]
            pv = candidate_peak_volumes[i]
            
            params = _fit_one(tseries, period, residual_metric, sp, pv, params)
            model = phoenix_r_with_period(params, tseries.shape[0])
            num_params = 5 * (i + 1) + 2
            curr_score = self.score_func(model, tseries, num_params, params)

            if (curr_score <= best_score):
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
