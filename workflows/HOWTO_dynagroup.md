We'll use a particular checkout of the repo, located on the HPC filesystem here:

/cluster/tufts/hugheslab/code/team-dynamics-time-series/

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
$ micromamba activate dynagroup_env_310
```

#### Step 3: Tell python where to find the dynagroup code

```
export PYTHONPATH=/cluster/tufts/hugheslab/code/team-dynamics-time-series/src/
```

### Step 3: Run code that demos performance on figure8

```
$ cd /cluster/tufts/hugheslab/code/team-dynamics-time-series/src/dynagroup/model2a/figure8/demos
$ python demo_cavi_on_figure8.py 
```
