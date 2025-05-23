# Environment setup instructions

We provide two options for environment setup, depending on which Python package manager you prefer to use: `micromamba` or `pyenv`. 

## Using micromamba

### 1) Install micromamba

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

### 2) Install a dedicated env for this project 

Uses file [`dynagroup_env.yml`](https://github.com/tufts-ml/team-dynamics-time-series/blob/mch_install/dynagroup_env.yml)

Found in the root of this git repo

```
$ cd $REPO_ROOT/
$ micromamba create -f dynagroup_env.yml
```

Will take about 5-10 min (on a laptop), somehow much longer on cluster

### 3) Activate and run a demo

```
$ micromamba activate dynagroup_env_310
$ cd $REPO_ROOT/src/dynagroup/model2a/figure8/demos/
$ python demo_cavi_on_figure8.py 
``` 

Should produce a lot of text to stdout and save lots of figures to files.


## Using pyenv

### 1) Create a dedicated env for this project
```
pyenv virtualenv 3.10.16 team
pyenv activate team
```

### 2) Install this project and dependencies
```
# install standard dependencies
pip install -r requirements.txt 
pip install -r dev-requirements.txt

# install Linderman's ssm dependency
pip install --no-build-isolation git+https://github.com/lindermanlab/ssm@6c856ad3967941d176eb348bcd490cfaaa08ba60 # no build isolation is because seem lacks pyproject.toml

# install this project 
pip install .
```

### 3) Run a demo

```
cd src/dynagroup/model2a/figure8/demos/
python demo_cavi_on_figure8.py 
``` 

### Appendix: On the `ssm` dependency

We must use additional care when installing the `ssm` dependency. In particular, we install it with the `--no-build-isolation` flag.   Why? As of PEP 517, setuptools runs setup.py in a clean build environment (which doesn’t include the existing virtual environment or its packages).  But building `ssm` requires a pre-existing package (`numpy`). Current best practice (as of PEP 518) is for a package in this situation to  include a `pyproject.toml` file, using `[build-system] requires` to declare that build dependencies (`numpy` in this case) must be available at build time.   However, `ssm` does not have a `pyproject.toml` file. We circumvent this deficiency of `ssm` by using the `--no-build-isolation` flag to allow `numpy` to exist in the build environment (since we would have already installed it in `requirements.txt`).