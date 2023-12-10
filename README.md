# CSBA-individual-assignment

This is my solution to the individual assignment of the Computer Science for Business Analytics course (EUR).

The problem is to detect duplicates in a large set (1624) of product descriptions of TVs from the web.
This is done through preselection, applying locality-sensitive hashing to the data in order to find candidate duplicates,
so as to reduce the cost of the subsequent duplicate detection through `...`.

I did this assignment individually.

## ToDo
- Refactor `solution.py` to be modular and use a `main` function
- Tune some of the parameters (shingle size, num_hash)
- Implement cross validation

#### Maybe
- Fix `Item` to be more consistent about return/setting own parameters

## Questions:
-

## Notes
- Filtering #occurrences of components <= 2 rather than == 1 yields worse results, too much loss of info

