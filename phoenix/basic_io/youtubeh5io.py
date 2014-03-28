#-*- coding: utf8
from __future__ import division, print_function

import numpy as np
import tables

tables.parameters.MAX_GROUP_WIDTH = 10e6
tables.parameters.IO_BUFFER_SIZE = 5 * 1024 * 1024
tables.parameters.DRIVER_DIRECT_BLOCK_SIZE = 16 * 1024
tables.parameters.NODE_CACHE_SLOTS = 0

video_ids_fpath = "/Users/flaviov/tseries/video_ids.txt"
h5fpath = "/Users/flaviov/tseries/tseries.h5"

def iter_videos():
    '''Iterates over video ids without the need to keep 3M strings in memory'''
    with open(video_ids_fpath) as video_ids_file:
        for line in video_ids_file:
            yield line.strip()

def get_good_videos():
    pops = []
    ages = []
    up_date = []
    good_series = []
    good_videos = []
    
    h5file = tables.open_file(h5fpath, "r")
    for video_id in iter_videos():
        views = h5file.get_node('/' + video_id + '/views').read()
        time_stamps = h5file.get_node('/' + video_id + '/time_stamps').read()
        age = time_stamps[1] - time_stamps[0]
        if views.shape[0] >= 30 and views.sum() > 500:
            good_series.append(views)
            good_videos.append(video_id)
            pops.append(views.sum())
            ages.append(age)
            up_date.append(time_stamps[0])
    
    h5file.close()
    return pops, ages, up_date, good_series, good_videos
