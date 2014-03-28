#-*- coding: utf8
from __future__ import division, print_function

import numpy as np
cimport numpy as np

cdef double[:] _shock(double beta, double gamma, double alpha, double r, \
        long s_0, long i_0, double d_t, double[:] store_at, Py_ssize_t start,
        Py_ssize_t end, int accumulate) nogil

cdef double[:] _phoenix_r(double[:, ::1] param_mat, double[:] store_at) nogil
