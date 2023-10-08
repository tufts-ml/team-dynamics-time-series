#!/bin/bash
#
# Usage
# -----
# $ bash launch_experiments.sh ACTION_NAME
#
# where ACTION_NAME is either 'list' or 'submit' or 'run_here'

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

export nLap1=60
export nLap2=10000
export nTask=20

export boxpath='/cluster/tufts/hugheslab/datasets/nba_cleveland_2015-16/v1.1.1.20231005/'
export output_path='/cluster/tufts/hugheslab/mhughe02/arhmm_results/'

for n_train_games in 1 5 20
do
    export n_train_games=$n_train_games

## Num states
for K in 04 08 16 32
do
    export K=$K

for hmmKappa in 500 2000
do
    export hmmKappa=$hmmKappa

## Scale of emission prior
for sF in 0.1 #0.02 0.5
do
    export sF=$sF

    ## NOTE all env vars that have been 'export'-ed will be passed along to the .slurm file

    echo "n_train=$n_train_games K=$K hmmKappa=$hmmKappa sF=$sF"

    if [[ $ACTION_NAME == 'submit' ]]; then
        ## Use this line to submit the experiment to the batch scheduler
        sbatch < run_train.slurm
    elif [[ $ACTION_NAME == 'run_first' ]]; then
        bash run_train.slurm
        exit
    elif [[ $ACTION_NAME == 'run_here' ]]; then
        ## Use this line to just run interactively
        bash run_train.slurm
    fi

done
done
done
done
