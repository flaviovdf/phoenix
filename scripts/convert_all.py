#!/usr/bin/env python
#-*- coding: utf8
from __future__ import division, print_function

from phoenix.basic_io import parser 

import os
import plac

BASE = '/Users/flaviov/tseries/user-item-pairs'

LASTFM_1K = os.path.join(BASE, 'lastfm-1k-users.txt')
HASHTAGS = os.path.join(BASE, 'hashtags-stanford-snap.txt')
ISMIR = os.path.join(BASE, 'ismir2013-mmusic-tweets.txt')

def main(output_fpath):
  
    parser.convert(ISMIR, output_fpath, 'ismir_artist', 
            parser.ismir, False)
    parser.convert(ISMIR, output_fpath, 'ismir_song', 
            parser.ismir, True)
    
    parser.convert(LASTFM_1K, output_fpath, 'lastfm_artist', 
            parser.lastfm, False)
    parser.convert(LASTFM_1K, output_fpath, 'lastfm_song', 
            parser.lastfm, True)

    parser.convert(HASHTAGS, output_fpath, 'twitter_hashtags', 
            parser.hashtag)

if __name__ == '__main__':
    plac.call(main)
