# CSBA-individual-assignment

This is my solution to the individual assignment of the Computer Science for Business Analytics course (EUR).

The problem is to detect duplicates in a large set (1624) of product descriptions of TVs from the web.
This is done through preselection, applying locality-sensitive hashing to the data in order to find candidate duplicates,
so as to reduce the cost of the subsequent duplicate detection through logistic regression with various similarity metrics as predictors.

I did this assignment individually.

## ToDo
- Train logit model on data and fix parameters rather than fitting each run
    - Can I adjust parameters for unbalancedness of training data? Doesn't really matter I guess
- Tune some of the parameters (logit parameters & threshold)
- Implement bootstrapping

#### Maybe
- Fix `Item` to be more consistent about return/setting own parameters

## Questions:
-

## Notes
- Filtering #occurrences of components <= 2 rather than == 1 yields worse results, too much loss of info

