shopt -s expand_aliases
#!/bin/bash

# https://unix.stackexchange.com/questions/1496/why-doesnt-my-bash-script-recognize-aliases
alias mpi1run='/usr/local/lib/openmpi-2.1.1/bin/mpirun -n 1'
alias mpi2run='/usr/local/lib/openmpi-2.1.1/bin/mpirun -n 2'
alias mpi4run='/usr/local/lib/openmpi-2.1.1/bin/mpirun -n 4'
alias mpi6run='/usr/local/lib/openmpi-2.1.1/bin/mpirun -n 6'
alias mpi8run='/usr/local/lib/openmpi-2.1.1/bin/mpirun -n 8'

var1=3
var2=2
incr=1
START=1
END=7

for ((i=START; i<=END; i++));
do
#    mpi1run python project/main.py
#    mpi2run python project/main.py
    mpi4run python project/main.py
    if (( i < END ));
    then
        sed -i "35s/$var1, $var2/$((var1+incr)), $((var2+incr))/" ./config/PABLO.ini
#        sed -i "30s/$var1/$((var1+incr))/" ./config/PABLO.ini
    fi
    var1=$((var1+incr))
    var2=$((var2+incr))
done

awk -v steps="$END" '{ sum += $1 } END { print "average time for " steps " refinements: " (sum/steps)  " secs."}' ./benchmark.txt
