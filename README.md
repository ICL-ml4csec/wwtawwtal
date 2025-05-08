# What We Talk About When We Talk About Logs
In our evaluations, endpoint threat detection systems are constantly achieving
essentially perfect performance, detecting all attacks with little to no false
positives. However, we don't believe these systems are ready for general
intrusion detection in practice. Then where do these misleading results come
from? We study the audit log datasets used to evaluate such systems in our
[paper](https://jasondliu.com/papers/sp25-wwtawwtal.pdf).

This repository contains the scripts we used to analyze our surveyed datasets.

## Setup
This code was run on an Ubuntu 20.04.6 LTS workstation using Python 3.8.10. We
recommend using a virtual environment to install dependencies:

```bash
$ python -m venv env --prompt wwtawwtal

$ # highly recommended: add `src` to PYTHONPATH
$ echo -e "\nexport PYTHONPATH=$PWD/src" >> env/bin/activate

$ source env/bin/activate
$ pip install -r requirements.txt
```

Note that you may need to pick a build of `pytorch` that specifically supports
your current workstation's CUDA version.

Various scripts are runnable, but they should all be called from `main.py` to
ensure classes are pickled correctly, e.g.:
```bash
$ python -m main neighborhood.neighborhood_nhop
```
will run the `main` function in `src/neighborhood/neighborhood_nhop.py`.
If the main needs arguments, they can be passed after the module name.

## About the scripts
This project is the culmination of years of research, including many
directions that didn't pan out very well. We left some of these previous
investigations in this repository for those who are interested. Older scripts
will generally not work out-of-the-box because data formats, etc., may have
changed since they were written.

Most of the final scripts used in the paper are found in `src/neighborhood`,
especially `neighborhood_nhop.py`, `rhoencoder.py`, and `plot_csv.py`.
and data conversions for the IDSes are found in `src/gnn`.
However, this repository doesn't include any of the IDSes that we evaluated.
These systems can be found from the releases of their corresponding papers.
Note that many of these IDSes need to be adapted to produce outputs for the
node classification setting that we use.
