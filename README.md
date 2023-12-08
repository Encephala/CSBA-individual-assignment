# CSBA-individual-assignment

This is my solution to the individual assignment of the Computer Science for Business Analytics course (EUR).

The problem is to detect duplicates in a large set (1624) of product descriptions of TVs from the web.
This is done through preselection, applying locality-sensitive hashing to the data in order to find candidate duplicates,
so as to reduce the cost of the subsequent duplicate detection through `...`.

I did this assignment individually.

## ToDo
- Better item representation than (shingles of) title
    - Find brand in title if not found in features (given set of known brands)
- Refactor `solution.py` to be modular and use a `main` function
- Tune some of the parameters (shingle size, num_hash)

#### Maybe
- Fix `Item` to be more consistent about return/setting own parameters

## Questions:
-

