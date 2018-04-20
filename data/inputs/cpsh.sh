#!/bin/bash
N=1000;
mkdir reducedval2017
for i in "./val2017"/*; do
    [ "$((N--))" = 0 ] && break
    cp -t "./reducedval2017" -- "$i"
done
