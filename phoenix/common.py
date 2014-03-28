# -*- coding: utf8
'''ContiguousID class'''
from __future__ import print_function, division

from collections import Mapping

class ContiguousID(Mapping):
    '''
    A ContiguousID is a dict in which keys map to integer values where values 
    are contiguous integers.
    
    Example:

    >>> x = ContiguousID()
    >>> x['a']
    0
    >>> x['b']
    1
    >>> x['a']
    0
    >>> x['c']
    2

    This class does not support setting items, values are automatic determined 
    when the item is first accessed.
    '''

    def __init__(self):
        '''
        Creates a new empty mapping.
        '''
        self.mem = {}
        self.reverse = {}
        self.curr_id = -1

    def __getitem__(self, key):
        '''
        Get's the value associated with the item. If the item has already been
        "looked up" it returns the previous key. Else, it will return the next
        integer beginning with 0. Example:

        >>> x = ContiguousID()
        >>> x['a']
        0
        >>> x['b']
        1
        >>> x['a']
        0
        >>> x[0]
        2
        '''

        if key in self.mem:
            return self.mem[key]
        else:
            self.curr_id += 1
            self.mem[key] = self.curr_id
            self.reverse[self.curr_id] = key
            return self.curr_id

    def boost(self, boost_val):
        '''
        Boost all ids by increment the `boost_val` param.
        
        Arguments
        ---------
        boost_val: int
            The value to boost by
        '''
        
        for key in self.mem.keys(): #This creates a copy, concurrent safe.
            self.mem[key] = self.mem[key] + boost_val
    
    def reverse_lookup(self, id_):
        return self.reverse[id_]

    def __iter__(self):
        return iter(self.mem)

    def __len__(self):
        return len(self.mem)
