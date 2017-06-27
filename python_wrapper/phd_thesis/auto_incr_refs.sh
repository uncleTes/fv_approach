shopt -s expand_aliases
#!/bin/bash

# https://unix.stackexchange.com/questions/1496/why-doesnt-my-bash-script-recognize-aliases
alias mpi2run='/usr/local/lib/openmpi-2.1.1/bin/mpirun -n 2'
alias mpi4run='/usr/local/lib/openmpi-2.1.1/bin/mpirun -n 4'
alias mpi6run='/usr/local/lib/openmpi-2.1.1/bin/mpirun -n 6'

var1=4
var2=5
incr=1
START=1
END=6

for ((i=START; i<=END; i++));
do
    mpi2run python project/main.py
    if (( i < END ));
    then
        sed -i "30s/$var1, $var2/$((var1+incr)), $((var2+incr))/" ./config/PABLO.ini
    fi
    var1=$((var1+incr))
    var2=$((var2+incr))
done

awk -v steps="$END" '{ sum += $1 } END { print "average time for " steps " refinements: " (sum/steps)  " secs."}' ./benchmark.txt