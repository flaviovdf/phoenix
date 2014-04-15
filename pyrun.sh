#!/bin/bash

function findself() {
    echo `dirname $0`
}

SELFDIR=`findself`

PYTHONPATH=$SELFDIR/ python $*

