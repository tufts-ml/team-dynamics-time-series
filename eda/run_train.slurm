#!/usr/bin/env bash
#SBATCH -n 1                # Number of nodes
#SBATCH -c 4                # Number of cores
#SBATCH -t 0-01:00          # Runtime in D-HH:MM
#SBATCH -p batch            # Partition to submit to
#SBATCH --mem-per-cpu 4000  # Memory (in MB) per cpu
#SBATCH -o logs/log_%j.out  # Write stdout to file named log_JOBIDNUM.out in current dir
#SBATCH -e logs/log_%j.err  # Write stderr to file named log_JOBIDNUM.err in current dir
#SBATCH --export=ALL        # Pass any exported env vars to this script and its children

# Be sure conda environment is loaded
HASBNPY=`python -c "import bnpy;print(1)"`
if [[ -z $HASBNPY ]]; then
    export PATH="/cluster/tufts/hugheslab/miniconda202108/bin:/cluster/tufts/hugheslab/miniconda202108/condabin:$PATH"
fi

python -c 'import sys; print(sys.path)'

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Python script reads in all args from ENV vars
python train_arhmm.py
