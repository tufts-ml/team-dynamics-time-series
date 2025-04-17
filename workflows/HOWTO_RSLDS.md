## Overview

We'll use the R-SLDS implementation in the ssm package maintained by lindermanlab

<https://github.com/lindermanlab/ssm/>

This is recommended over the older recurrent-slds package, which has more installation issues when attempted in Apr '25

#### Usage examples

For usage of R-SLDS, see:

* <https://github.com/lindermanlab/ssm/tree/master/examples>
* <https://github.com/lindermanlab/ssm/tree/master/notebooks>

## Howto steps

#### Step 0: Log in to HPC cluster

#### Step 1: Configure your environment via env variables

```
export PATH="/cluster/tufts/hugheslab/micromamba/bin/:$PATH"
export MAMBA_ROOT_PREFIX="/cluster/tufts/hugheslab/micromamba/"
export MAMBA_EXE="/cluster/tufts/hugheslab/micromamba/bin/micromamba"
```

#### Step 2: Activate the ssm_env

```
$ micromamba activate ssm_env
```

### Step 3: Run code

```
$ cd /cluster/tufts/hugheslab/code/ssm/examples/
$ python rslds.py
```
