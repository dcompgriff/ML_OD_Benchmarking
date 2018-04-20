#!/bin/bash
for j in 0 1 2 3 4 5 6 7; do
    N=125
    mkdir input$j
    for i in "./reducedval2017"/*; do
        [ "$((N--))" = 0 ] && break
        mv -t "./input$j" -- "$i"
    done
done
