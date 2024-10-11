#!/bin/bash

main(){
    setup=/home/mcd4/Documents/metavision/openeb/build/utils/scripts/setup_env.sh
    py=/home/mcd4/miniconda3/envs/openeb/bin/python
    file_name=_fusion_executable_.py
    modes=( "cytovia" "evanescent" )
    conditions=( 0 1 2 3 4 5 6 7 8 )
    for condition in ${conditions[*]}
    do
        for mode in ${modes[*]}
        do
            echo "MODE = $mode; condition = $condition"
            $setup $py $file_name --mode $mode --condition_number $condition &
        done
        wait
    done
}

main
