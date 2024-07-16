#!/bin/bash

main(){
    setup=/home/mcd4/Documents/metavision/openeb/build/utils/scripts/setup_env.sh
    py=/home/mcd4/miniconda3/envs/openeb/bin/python
    file_name=_fusion_executable_.py
    n_proc=3
    $setup mpiexec -n $n_proc $py $file_name
}

main
