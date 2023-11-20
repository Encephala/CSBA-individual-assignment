# CSBA-individual-assignment

This is my solution to the individual assignment of the Computer Science for Business Analytics course (EUR).

The problem is to detect duplicates in a large set (1624) of product descriptions of TVs from the web.
This is done through preselection, applying locality-sensitive hashing to the data in order to find candidate duplicates,
so as to reduce the cost of the subsequent duplicate detection through the multi-component similarity method.

I did this assignment individually.

## ToDo
- Implement duplicate checking given the candidates
- Implement performance evaluation
- Tune some of the parameters (shingle size, n, b/r)

## Questions:
- So about 50% F1-score and 4% F1*-score after MSM, but any indication what the scores were before MSM?

