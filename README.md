# dynagroup

Welcome to dynagroup


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





