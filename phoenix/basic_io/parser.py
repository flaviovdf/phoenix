#-*- coding: utf8
from __future__ import division, print_function

from phoenix.common import ContiguousID

import tables
import time

class UserObject(tables.IsDescription):
    user_id = tables.Int32Col()
    object_id = tables.Int32Col()
    date = tables.Int32Col()

def hashtag(input_fpath):

    with open(input_fpath) as input_file:
        for line in input_file:
            spl = line.split()
            
            user = spl[0]
            hashtag = spl[1]
            date = ' '.join(spl[2:])
            
            date_epoch = time.mktime(
                    time.strptime(date, '%Y-%m-%d %H:%M:%S'))
            
            yield user, hashtag, date_epoch

def ismir(input_fpath, use_tracks=False):
    
    if use_tracks:
        object_col = 4
    else:
        object_col = 3

    with open(input_fpath) as input_file:
        first_line = True
        for line in input_file:
            if first_line:
                first_line = False
                continue

            spl = line.split()
            
            user = spl[2]
            obj = spl[object_col]

            date = ' '.join(spl[5:7])
            date_epoch = time.mktime(
                    time.strptime(date, '%Y-%m-%d %H:%M:%S'))

            yield user, obj, date_epoch

def lastfm(input_fpath, use_tracks=False):

    if use_tracks:
        object_col = 4
    else:
        object_col = 2

    with open(input_fpath) as input_file:
        for line in input_file:
            spl = line.split('\t')

            user = spl[0]
            date_aux = spl[1]
            obj = spl[object_col]
            
            if obj.strip() == '':
                continue

            date_spl = date_aux.split('T')
            date_day = date_spl[0]
            date_hour = date_spl[1][:-1]
            date = '%s %s' % (date_day, date_hour)

            date_epoch = time.mktime(
                    time.strptime(date, '%Y-%m-%d %H:%M:%S'))

            yield user, obj, date_epoch

def convert(input_fpath, table_fpath, table_name, converter, *args):

    user_ids = ContiguousID()
    object_ids = ContiguousID()

    h5_file = tables.open_file(table_fpath, 'a')
    table = h5_file.create_table('/', table_name, UserObject)
    
    for user, obj, date in converter(input_fpath, *args):
        row = table.row

        user_id = user_ids[user]
        object_id = object_ids[obj]

        row['user_id'] = user_id
        row['object_id'] = object_id
        row['date'] = date
        row.append()

    table.flush()
    table.cols.date.create_csindex()
    h5_file.close()
