# CSBA-individual-assignment

[![CFF valid](https://github.com/Encephala/CSBA-individual-assignment/actions/workflows/validate_cff.yaml/badge.svg)](https://github.com/Encephala/CSBA-individual-assignment/actions/workflows/validate_cff.yaml)

This is my solution to the individual assignment of the Computer Science for Business Analytics course (EUR).

The problem is to detect duplicates in a large set (1624) of product descriptions of TVs from the web.
This is done through preselection, applying locality-sensitive hashing to the data in order to find candidate duplicates,
so as to reduce the cost of the subsequent duplicate detection through logistic regression with various similarity metrics as predictors.

A final F1 score of 27% is achieved when filtering out 85% of the potential duplicates with LSH. Further explanation can be found in the accompanying paper, `Paper.pdf`.

I did this assignment individually.
