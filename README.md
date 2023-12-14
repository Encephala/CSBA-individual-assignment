# CSBA-individual-assignment

[![CFF valid](https://github.com/Encephala/CSBA-individual-assignment/actions/workflows/validate_cff.yaml/badge.svg)](https://github.com/Encephala/CSBA-individual-assignment/actions/workflows/validate_cff.yaml)

This is my solution to the individual assignment of the Computer Science for Business Analytics course (EUR).

The problem is to detect duplicates in a large set (1624) of product descriptions of TVs from the web.
This is done through preselection, applying locality-sensitive hashing to the data in order to find candidate duplicates,
so as to reduce the cost of the subsequent duplicate detection through logistic regression with various similarity metrics as predictors.

I did this assignment individually.

## ToDo
- Tune logit threshold
- Implement some way to aggregate results across bootstraps

## Questions:
-

## Notes
- Filtering #occurrences of components <= 2 rather than == 1 yields worse results, too much loss of info
- Intercept may have to be adjusted for the unbalancedness of the data

