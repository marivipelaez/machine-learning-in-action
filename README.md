
# Examples from Machine Learning in Action

* [More info on machine learning in action](https://github.com/pbharrin/machinelearninginaction)

## Dependencies
The project is implemented with Python 2.7.9 and, as recommended in the book, 
using the following libraries (see requirements.txt):
* numpy
* scipy
* ipython
* matplotlib
* seaborn
* pandas
* shapely

To avoid mixing things up, please use a virtualenv to isolate the project execution environment.

## Repository content
There is a package per book chapter. Their names consist of 'ch{chapter}_{title}', in lower.
Data is not included in the repository, as I'm using directly the datasets provided in the examples,
that could be found in [the book github repository](https://github.com/pbharrin/machinelearninginaction).

Dependencies of the code are in requirements.txt file. This file could be used with pip to download all
the packages:

```shell
$ pip install -r requirements.txt
```

## AOT

* [Installing all ipython dependencies](http://ipython.org/ipython-doc/dev/install/install.html#installnotebook)

```shell
(venv)$ pip install ipython[all]
```

Then, access project notebooks from: http://localhost:8888/notebooks

* Reloading modules in ipython:

```python
%dreload(already_imported_module)
```

