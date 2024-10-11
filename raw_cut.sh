#!/bin/bash

#-------------------------------------------------------------------------------
#         Wrapper for cutting the raw files to time sync them...
#-------------------------------------------------------------------------------

main(){
    exe=metavision_file_cutter
    files=$(find time_synced -name "*.csv")

    for f in ${files[*]}
    do
        parent_dir=$(dirname $f)
        path_local=$(grep "path_to_raw" $f | awk '{print $2}')
        time_init=$(grep "time_init" $f | awk '{print $2}')
        time_end=$(grep "time_end" $f | awk '{print $2}')
        echo $parent_dir
        echo $path_local
        echo $time_init
        echo $time_end
        $exe -i $path_local \
             -o $parent_dir/raw_time_cut.raw \
             -s $time_init \
             -e $time_end
    done
}

main
