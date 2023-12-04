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

Here we provide scripts for reproducing experiments on publicly available data (_FigureEight_, basketball).

Baseline forecasts for non-ablation models can be obtained via the following external repos:

1. [_FigureEight_: DSARF](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm)
2. [Basketball: AgentFormer](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm) 
3. [Basketball: GroupNet](https://github.com/mikewojnowicz/GroupNet/tree/aistats)

To train our model (HSRDM) as well as various ablations, use

1. [_FigureEight_: HSRDM, rAR-HMM](src/dynagroup/model2a/figure8/demos/demo_cavi_on_figure8.py)
2. [Basketball: HSRDM, rAR-HMM, no-system-state ablation](src/dynagroup/model2a/basketball/demos/baller2vec_format/CLE_starters/demo_full_pipeline.py)

To show results, use

1. [_FigureEight_: DSARF](src/dynagroup/model2a/figure8/diagnostics/plot_external_forecasts.py) (results from our model constructed via above script)
2. [Basketball: all models](/Users/miw267/Repos/dynagroup/src/dynagroup/model2a/basketball/demos/baller2vec_format/CLE_starters/demo_metrics_from_forecast_collection.py)







