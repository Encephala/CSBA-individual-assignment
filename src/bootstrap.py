#!/usr/bin/env python3

from random import choice
from datetime import datetime

import numpy as np

from solution import *

num_hashes = 432
weight = 1.0

# Drop highest divisors, as they all yield 0 TP after LSH anyways
all_divisors = [i for i in range(1, num_hashes + 1) if num_hashes % i == 0][:8]
print(f"Testing for num_rows in {all_divisors}")


filename = "data/TVs-all-merged.json"
products, all_duplicates, num_products = load_data(filename)

do_print = False

num_bootstraps = 5

# results[:, :, i] is:
# i = 0 - comparison ratio
# i = 1 - precision_star
# i = 2 - recall_star
# i = 3 - F1_star
# i = 4 - precision
# i = 5 - recall
# i = 6 - F1
results = np.empty([num_bootstraps, len(all_divisors), 7])


for bootstrap in range(num_bootstraps):
    print(f"Bootstrap {bootstrap + 1} / {num_bootstraps}")

    bootstrap_train: set[Item] = set()

    for _ in range(len(products)):
        bootstrap_train.add(choice(tuple(products)))


    bootstrap_test: list[Item] = list(set(products) - bootstrap_train)
    bootstrap_train: list[Item] = list(bootstrap_train)

    all_duplicates_train: set[tuple[Item, Item]] = set()
    all_duplicates_test: set[tuple[Item, Item]] = set()

    for i, product in enumerate(bootstrap_train):
        for other_product in bootstrap_train[i + 1:]:
            if (product, other_product) in all_duplicates or (other_product, product) in all_duplicates:
                all_duplicates_train.add((product, other_product))

    for i, product in enumerate(bootstrap_test):
        for other_product in bootstrap_test[i + 1:]:
            if (product, other_product) in all_duplicates or (other_product, product) in all_duplicates:
                all_duplicates_test.add((product, other_product))

    num_products_train = len(bootstrap_train)
    num_products_test = len(bootstrap_test)

    print(f"Train: {num_products_train}, test: {num_products_test}")
    print(f"All duplicates train: {len(all_duplicates_train)}, all duplicates test: {len(all_duplicates_test)}", end = "\n\n")

    for (divisor_index, num_rows) in enumerate(all_divisors):
        print(f"{divisor_index + 1} / {len(all_divisors)}: {num_rows} rows")

        num_bands = num_hashes // num_rows

        if do_print:
            print(f"(Approximate) LSH Acceptance threshold: {(1 / num_bands) ** (1 / num_rows):.4f}")


        # Fit model on train data
        if do_print:
            print()
            print("Train data")

        signatures_train = minhash(bootstrap_train, num_hashes, do_print = do_print)

        intermediate_duplicates_train = LSH(bootstrap_train, signatures_train, num_bands, num_rows)

        comparison_ratio_train = len(intermediate_duplicates_train) / comb(len(bootstrap_train), 2)

        if do_print:
            print(f"Comparison ratio: {comparison_ratio_train:.1%}")

        precision_star, recall_star, F1_star = evaluate(intermediate_duplicates_train, all_duplicates_train, num_products_train, do_print = do_print)

        # Can't do logit if we have 0 TP in training data
        if precision_star != 0:
            _, predictor = duplicate_detection(intermediate_duplicates_train, all_duplicates_train, weight = weight, do_print = do_print)

        else:
            results[bootstrap][divisor_index] = [comparison_ratio_train, 0, 0, 0, 0, 0, 0]
            continue


        # Apply model to testing data
        if do_print:
            print()
            print("Test data")

        signatures_test = minhash(bootstrap_test, num_hashes, do_print = do_print)

        intermediate_duplicates_test = LSH(bootstrap_test, signatures_test, num_bands, num_rows)

        precision_star, recall_star, F1_star = evaluate(intermediate_duplicates_test, all_duplicates_test, num_products_test, do_print = do_print)

        comparison_ratio_test = len(intermediate_duplicates_test) / comb(len(bootstrap_test), 2)

        if do_print:
            print(f"Comparison ratio: {comparison_ratio_test:.1%}")

        final_duplicates, _ = duplicate_detection(intermediate_duplicates_test, all_duplicates_test, predictor = predictor, do_print = do_print)

        precision, recall, F1 = evaluate(final_duplicates, all_duplicates_test, num_products_test, do_print = do_print)


        results[bootstrap][divisor_index] = [comparison_ratio_test, precision_star, recall_star, F1_star, precision, recall, F1]

        if do_print:
            print()

    print()

np.save(f"data/bootstrap {datetime.now()}-{all_divisors}.npy", results)
