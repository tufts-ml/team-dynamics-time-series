#!/bin/bash

for domask in 1 0
do
for seed in 101 202 303 404 505 606 707 808 909
do
    python -W ignore::UserWarning run_hsrdm.py --seed $seed --n_cavi_iterations 20 --seeds_for_forecasting "" --verbose 1 --init_n_em_iters_bottom 5 --init_n_em_iters_top 5 --use_provided_data_mask $domask
done
done
