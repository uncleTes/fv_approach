#!/bin/bash

for i in `seq 1 10`;
do
    mpirun -n 4 python project/main.py
done

awk '{ sum += $1 } END { print "average time for 10 repetitions: " (sum/10)  " secs."}' ./benchmark.txt
