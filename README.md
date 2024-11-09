# dynagroup

Welcome to `dynagroup`.   This is a Python repo for the under-review paper _Discovering group dynamics in synchronous time series via hierarchical recurrent switching-state models_.

# Installation

The package currently only works on python3.8 (or possibly below). It does not work out-of-the-box on python 3.9 or python 3.10, due to issues inherited from dependencies.

Installation (to a virtual environment, using pyenv) can be done as follows:

```
pyenv virtualenv 3.8.18 env
pip install -e . 
pip install -r dev-requirements.txt
```
where we have assumed that pyenv has already installed python 3.8.18 to the computer.   

These commands install a virtual environment with python 3.8.18, editably install `dynagroup`, and then install development requirements (such as `pytest`, used for running unit tests).   The last step can be skipped if desired. 

# Unit tests


Unit tests can be run from within the activated virtual environment using 

```
python -m python
```

# Experiment reproduction

Here we provide scripts for reproducing experiments on publicly available data (_FigureEight_, Basketball, and MarchingBand).

A. Baseline forecasts for non-ablation models can be obtained via the following external repos:

1. [_FigureEight_: DSARF](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm/tree/kgili/dsarf_on_figure_8)
2. [Basketball: AgentFormer](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm/agentformer_on_bball) 
3. [Basketball: GroupNet](https://github.com/mikewojnowicz/GroupNet/tree/aistats)
4. [MarchingBand: DSARF](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm/tree/kgili/dsarf_on_marching)

B. To train our model (HSRDM) as well as various ablations, use: 

1. [_FigureEight_: HSRDM, rAR-HMM](https://github.com/tufts-ml/team-dynamics-time-series/tree/kgdev/src/dynagroup/model2a/figure8/demos/demo_cavi_on_figure8.py)
2. [_FigureEight_: HSRDM, rAR-HMM Pool.](https://github.com/tufts-ml/team-dynamics-time-series/tree/kgdev/src/dynagroup/model2a/figure8/demos/demo_fig8_complete_pooling.py)
3. [_FigureEight_: HSRDM, rAR-HMM Concat.](https://github.com/tufts-ml/team-dynamics-time-series/tree/kgdev/src/dynagroup/model2a/figure8/demos/demo_fig8_concatenation.py)
4. [Basketball: HSRDM, rAR-HMM, no-recurrence ablation](src/dynagroup/model2a/basketball/demos/baller2vec_format/CLE_starters/demo_full_pipeline.py)
5. [MarchingBand: HSRDM, rAR-HMM, no-recurrence ablation](https://github.com/tufts-ml/team-dynamics-time-series/tree/kgdev/src/dynagroup/model2a/marching_band/demo.py)

C. To train baseline forecasts, use: 

1. [_FigureEight_: DSARF Ind., Pool., Concat.](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm/tree/kgili/dsarf_on_figure_8/DSARF_on_figure_8.ipynb)
2. [Basketball: AgentFormer](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm/agentformer_on_bball) 
3. [Basketball: GroupNet](https://github.com/mikewojnowicz/GroupNet/tree/aistats)
4. [MarchingBand: DSARF](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm/tree/kgili/dsarf_on_marching/run_marchingband.py)


C. Experiments from the paper can be reproduced by running the scripts/notebooks above. For exact reproducibility, exact hyperparameters and seeds need to be used. Results will either be automatically plotted and saved in the following folders, or can be easily reproduced in the following Jupyter Notebooks: 

1. [_FigureEight_: HSRDM, rAR-HMM](https://github.com/tufts-ml/team-dynamics-time-series/tree/kgdev/src/dynagroup/model2a/figure8/demos/results/plots)
2. [_FigureEight_: HSRDM, rAR-HMM Pool.](https://github.com/tufts-ml/team-dynamics-time-series/tree/kgdev/src/dynagroup/model2a/figure8/demos/results/plots)
3. [_FigureEight_: HSRDM, rAR-HMM Concat.](https://github.com/tufts-ml/team-dynamics-time-series/tree/kgdev/src/dynagroup/model2a/figure8/demos/results/plots)
4. [_FigureEight_: DSARF Ind., Pool., Concat.](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm/tree/kgili/dsarf_on_figure_8)
5. [Basketball: HSRDM, rAR-HMM, no-recurrence ablation](src/dynagroup/model2a/basketball/demos/baller2vec_format/CLE_starters/demo_full_pipeline.py)
6. [MarchingBand: HSRDM, rAR-HMM, no-recurrence ablation](https://github.com/tufts-ml/team-dynamics-time-series/tree/kgdev/src/dynagroup/model2a/marching_band/results/plots.py)
7. [MarchingBand: DSARF Concat.](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm/tree/kgili/dsarf_on_marching/plot_results.ipynb)




