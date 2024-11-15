# dynagroup

Welcome to `dynagroup`. This is a Python repo for the under-review paper _Discovering group dynamics in synchronous time series via hierarchical recurrent switching-state models_.

## Installation

See [INSTALL.md](#INSTALL.md) to setup this project's micromamba environment.

You'll have an environment with Python 3.10 called `dynagroup_env_310`

## Verifying install

Unit tests can be run from within the activated virtual environment using 

```
python -m pytest
```

## Experiment reproduction

Here we provide scripts for reproducing experiments on publicly available data (_FigureEight_, basketball).

A. Baseline forecasts for non-ablation models can be obtained via the following external repos:

1. [_FigureEight_: DSARF](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm)
2. [Basketball: AgentFormer](https://github.com/tufts-ml/dsarf_agentformer_baseline_for_hsrdm) 
3. [Basketball: GroupNet](https://github.com/mikewojnowicz/GroupNet/tree/aistats)

B. To train our model (HSRDM) as well as various ablations, use

1. [_FigureEight_: HSRDM, rAR-HMM](src/dynagroup/model2a/figure8/demos/demo_cavi_on_figure8.py)
2. [Basketball: HSRDM, rAR-HMM, no-system-state ablation](src/dynagroup/model2a/basketball/demos/baller2vec_format/CLE_starters/demo_full_pipeline.py)

C. To show results, use

1. [_FigureEight_: DSARF](src/dynagroup/model2a/figure8/diagnostics/plot_external_forecasts.py) (results from our model constructed via above script)
2. [Basketball: all models](src/dynagroup/model2a/basketball/demos/baller2vec_format/CLE_starters/demo_metrics_from_forecast_collection.py). Note that this script requires locating the forecasts that were written to disk in step B1. 






