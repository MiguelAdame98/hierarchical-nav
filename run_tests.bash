#!/bin/bash

echo "Run model in 2x3 rooms environments with the desired variables"

declare -a ENVS=("4-tiles-ad-rooms" "5-tiles-ad-rooms")
declare -a LOOKAHEADS=("6" "7")


#TEST 0 : keyboard
#test 1: exploration
#test 2: 
export START_SEED=200
export END_SEED=213 
export TEST=$1
export ROW=$2 
export COL=$3 
export Saving_dir=check_refactoring

export GPULAB_dir=/project_ghent/ddtinguy/hierarchical_st_nav_aif/

echo Test $TEST
echo Saving dir $Saving_dir


for env_setting in $(seq 0 0)
	do
	ENV=${ENVS[$env_setting]}
	LOOKAHEAD=${LOOKAHEADS[$env_setting]}
	
	echo $ENV
    
    for seed in $(seq $START_SEED $END_SEED)
        do

        echo seed= $seed
        echo ROW= $ROW
        echo col= $COL
        #work here
        n_tile=${ENV:0:1}

        folder=${GPULAB_dir}${Saving_dir}/${n_tile}t_${ROW}x${COL}_s${seed} 
        mkdir -p $folder
        file=${TEST}_${ENV}_${ROW}x${COL}_s${seed}.txt
        echo saving logs in ${folder}/${file}
        python run.py --allo_config ${GPULAB_dir}runs/GQN_V2_AD/v2_GQN_AD_conv7x7_bi/GQN.yml --env $ENV --seed $seed --rooms_in_row $ROW --rooms_in_col $COL --test $TEST --video --save_dir $Saving_dir --lookahead $LOOKAHEAD >> ${folder}/${file}
        
    done #seed
            
done #envs
exit



