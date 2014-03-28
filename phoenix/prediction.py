#-*- coding: utf8
from __future__ import division, print_function

from phoenix.models import WaveFitPhoenixR
from phoenix.models import FixedParamsPhoenixR

from itertools import product

import numpy as np

def cid(x, y):
    
    ce_x = np.sqrt((np.ediff1d(x) ** 2).sum())
    ce_y = np.sqrt((np.ediff1d(y) ** 2).sum())

    ed = np.sqrt(((x - y) ** 2).sum())
    cf = max(ce_x, ce_y) / min(ce_x, ce_y)

    return ed * cf

def cid_all(X, Y):

    ce_x = np.sqrt((np.diff(X) ** 2).sum(axis=1))
    ce_y = np.sqrt((np.diff(Y) ** 2).sum(axis=1))

    ED = sp_dist.cdist(X, Y)

    R = np.zeros((X.shape[0], Y.shape[0]), dtype='f')
    for i, j in product(xrange(X.shape[0]), xrange(Y.shape[0])):
        cf = max(ce_x[i], ce_y[j]) / min(ce_x[i], ce_y[j])
        cid_dist = ED[i, j] * cf
        R[i, j] = cid_dist
    
    return R

def predict_one(x, C, k):

    dists = cid_all(x[None], C)[0]
    
    closest = dists.argsort()[:k]
    

    for i in C.shape[0]:


    init_params = average_params(models_dict, closest)
