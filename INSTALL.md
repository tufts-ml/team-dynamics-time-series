How MCH got things to install on both Tufts HPC and his laptop

# 1) Install micromamba

Follow these directions

<https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html>

Usually, just as simple as

```
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

NB: on Tufts cluster, just use

```
export PATH="/cluster/tufts/hugheslab/micromamba/bin/:$PATH"
```

Otherwise, on your laptop, be sure your path is updated

```
export PATH="/path/to/bin/micromamba:$PATH"
```

# 2) Install a dedicated env for this project 

Uses file `dynagroup_env.yml`

```
$ cd $REPO_ROOT/
$ micromamba create -f dynagroup_env.yml
```

Will take about 5-10 min (on a laptop), somehow much longer on cluster

# 3) Activate and run a test

```
$ micromamba activate dynagroup_env_310
$ cd $REPO_ROOT/src/dynagroup/model2a/figure8/demos/
$ python demo_cavi_on_figure8.py 
``` 

Should produce a lot of text to stdout and save lots of figures to files.
