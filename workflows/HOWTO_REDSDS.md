## Overview

We'll use the REDSDS implementation provided by that work's authors:

<https://github.com/abdulfatir/REDSDS>

#### Usage examples

For usage for simple forecasting, see:

* <https://github.com/abdulfatir/REDSDS/blob/main/run_gts_univariate.py>

## Howto steps

#### Step 0: Log in to HPC cluster

#### Step 1: Configure your environment via env variables

```
export PATH="/cluster/tufts/hugheslab/micromamba/bin/:$PATH"
export MAMBA_ROOT_PREFIX="/cluster/tufts/hugheslab/micromamba/"
export MAMBA_EXE="/cluster/tufts/hugheslab/micromamba/bin/micromamba"
```

#### Step 2: Activate the environment

```
$ micromamba activate redsds_env
```

### Step 3: Run code

```
$ cd /cluster/tufts/hugheslab/code/REDSDS/
$ python run_gts_univariate.py --config configs/traffic_duration.yaml --device cuda:0
```
