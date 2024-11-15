# Environment setup instructions

## 1) Install micromamba

Follow these directions

<https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>

Usually, just as simple as

```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

NB: on an HPC cluster, you can try to use an existing install:

```
export PATH="/cluster/tufts/hugheslab/micromamba/bin/:$PATH"
```

Otherwise, on your laptop, be sure your path is updated

```
export PATH="/path/to/bin/micromamba:$PATH"
```

## 2) Install a dedicated env for this project 

Uses file [`dynagroup_env.yml`](https://github.com/tufts-ml/team-dynamics-time-series/blob/mch_install/dynagroup_env.yml)

Found in the root of this git repo

```
$ cd $REPO_ROOT/
$ micromamba create -f dynagroup_env.yml
```

Will take about 5-10 min (on a laptop), somehow much longer on cluster

## 3) Activate and run a test

```
$ micromamba activate dynagroup_env_310
$ cd $REPO_ROOT/src/dynagroup/model2a/figure8/demos/
$ python demo_cavi_on_figure8.py 
``` 

Should produce a lot of text to stdout and save lots of figures to files.
