# Multiple testing for signal-agnostic searches for new physics with machine learning

This is the code to reproduce the results of the paper

```
@article{Grosso:2024wjt,
    author = "Grosso, Gaia and Letizia, Marco",
    title = "{Multiple testing for signal-agnostic searches of new physics with machine learning}",
    eprint = "2408.12296",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "8",
    year = "2024"
}
```
The aggregation methods are defined in the utils.py file.

The notebook [powers](powers.ipynb) contains the computation of the results of the aggregated tests from the individual tests in the test_statistics folder.
The notebook [correlations](correlations.ipynb) shows the degree of correlation among the base tests, each defined by a different value of the kernel width.

Contributors: Gaia Grosso and Marco Letizia
