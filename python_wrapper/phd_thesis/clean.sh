#!/bin/bash

CURRENT_DIR_PATH=$(pwd)
EXECUTE_DIR_PATH="/home/federico/WorkSpace/PythonProjects/PhdThesis/bitpit1.2/bitpit-1.2/python_wrapper/phd_thesis"

if [ "$EXECUTE_DIR_PATH" == "$CURRENT_DIR_PATH" ]; then
    [ -f ./benchmark.txt ] && rm -v ./benchmark.txt

    count=`ls -1 ./*.lprof 2>/dev/null | wc -l`

    if [ $count != 0 ]; then 
        rm -v ./*.lprof
    fi

    SUB_DIR="./data"
    
    if [ -d "$SUB_DIR" ]; then

        count=`ls -1 ./data/*.vtu 2>/dev/null | wc -l`

        if [ $count != 0 ]; then 
            rm -v ./data/*.vtu
        fi

        count=`ls -1 ./data/*.pvtu 2>/dev/null | wc -l`

        if [ $count != 0 ]; then 
            rm -v ./data/*.pvtu
        fi 

        [ -f ./data/multiple_PABLO.vtm ] && rm -v ./data/multiple_PABLO.vtm
        
    fi

    SUB_DIR="./log"
    
    if [ -d "$SUB_DIR" ]; then

        count=`ls -1 ./log/*.log 2>/dev/null | wc -l`

        if [ $count != 0 ]; then 
            rm -v ./log/*.log
        fi

        cd ../
    fi

else
    echo "Execution path must be $EXECUTE_DIR_PATH"
fi
    
