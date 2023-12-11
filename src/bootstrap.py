#!/usr/bin/env python3

import matplotlib.pyplot as plt

from dataclasses import dataclass
from random import choice

from solution import *

num_hashes = 120 # TODO remove after testing

all_divisors = [i for i in range(1, num_hashes + 1) if num_hashes % i == 0]
all_divisors = [3, 4, 5] # TODO remove after testing

@dataclass
class Result():
    precision_star: float
    recall_star: float
    F1_star: float

    precision: float
    recall: float
    F1: float


filename = "data/TVs-all-merged.json"
products, all_duplicates, num_products = load_data(filename)

do_print = False

num_bootstraps = 5

results: list[dict[float, Result]] = [{} for _ in range(num_bootstraps)]

for bootstrap in range(num_bootstraps):
    print(f"Bootstrap {bootstrap + 1} / {num_bootstraps}")

    bootstrap_train = set()

    for _ in range(len(products)):
        bootstrap_train.add(choice(tuple(products)))


    bootstrap_test = list(set(products) - bootstrap_train)
    bootstrap_train = list(bootstrap_train)

    num_products_train = len(bootstrap_train)
    num_products_test = len(bootstrap_test)

    print(f"Train: {num_products_train}, test: {num_products_test}", end = "\n\n")


    for (i, num_rows) in enumerate(all_divisors):
        print(f"{i + 1} / {len(all_divisors)}: {num_rows} rows")

        num_bands = num_hashes // num_rows
        # Check that num_hashes is divisible by num_rows
        assert num_bands * num_rows == num_hashes

        if do_print:
            print(f"(Approximate) LSH Acceptance threshold: {(1 / num_bands) ** (1 / num_rows):.4f}")


        # Fit model on train data
        minhash(bootstrap_train, num_hashes, do_print)

        intermediate_duplicates = LSH(bootstrap_train, num_bands, num_rows)

        comparison_ratio = len(intermediate_duplicates) / comb(len(bootstrap_train), 2)

        if do_print:
            print(f"Comparison ratio: {comparison_ratio:.1%}")

        precision_star, recall_star, F1_star = evaluate(intermediate_duplicates, all_duplicates, num_products_train, do_print)

        # Can't do logit if we have 0 TP in training data
        if precision_star != 0:
            final_duplicates, predictor = duplicate_detection(intermediate_duplicates, all_duplicates, do_print = do_print)

        else:
            results[bootstrap][comparison_ratio] = Result(0, 0, 0, 0, 0, 0)
            continue


        # Apply model to testing data
        minhash(bootstrap_test, num_hashes, do_print)

        intermediate_duplicates = LSH(bootstrap_test, num_bands, num_rows)

        precision_star, recall_star, F1_star = evaluate(intermediate_duplicates, all_duplicates, num_products_test, do_print)

        comparison_ratio = len(intermediate_duplicates) / comb(len(bootstrap_test), 2)

        if do_print:
            print(f"Comparison ratio: {comparison_ratio:.1%}")

        final_duplicates, _ = duplicate_detection(intermediate_duplicates, all_duplicates, predictor, do_print)

        precision, recall, F1 = evaluate(final_duplicates, all_duplicates, num_products_test, do_print)


        results[bootstrap][comparison_ratio] = Result(precision_star, recall_star, F1_star, precision, recall, F1)

        if do_print:
            print()

    print()


# Average results across bootstraps
averaged_results = {}

for result in results:
    print(result.keys())

# This doesn't work, as the comparison ratio isn't constant across bootstraps
for comparison_ratio in results[0].keys():
    precision_star = sum(result[comparison_ratio].precision_star for result in results) / num_bootstraps
    recall_star = sum(result[comparison_ratio].recall_star for result in results) / num_bootstraps
    F1_star = sum(result[comparison_ratio].F1_star for result in results) / num_bootstraps

    precision = sum(result[comparison_ratio].precision for result in results) / num_bootstraps
    recall = sum(result[comparison_ratio].recall for result in results) / num_bootstraps
    F1 = sum(result[comparison_ratio].F1 for result in results) / num_bootstraps

    averaged_results[comparison_ratio] = Result(precision_star, recall_star, F1_star, precision, recall, F1)

key = results[0].keys()[0]
print(results[0][key])
print(averaged_results[key])

comparison_ratios = [100 * i for i in averaged_results.keys()]

plt.figure()
plt.title("Performance LSH")
plt.xlabel("Fraction of comparisons (%)")
plt.ylabel("Performance (%)")
plt.plot(comparison_ratios, [result.precision_star * 100 for result in results.values()], label = "Precision*")
plt.plot(comparison_ratios, [result.recall_star * 100 for result in results.values()], label = "Recall*")
plt.plot(comparison_ratios, [result.F1_star * 100 for result in results.values()], label = "F1*")
plt.legend()
plt.savefig("images/performance_LSH.png")
plt.savefig("images/performance_LSH.svg")

plt.figure()
plt.title("Performance final")
plt.xlabel("Fraction of comparisons (%)")
plt.ylabel("Performance (%)")
plt.plot(comparison_ratios, [result.precision * 100 for result in results.values()], label = "Precision")
plt.plot(comparison_ratios, [result.recall * 100 for result in results.values()], label = "Recall")
plt.plot(comparison_ratios, [result.F1 * 100 for result in results.values()], label = "F1")
plt.legend()
plt.savefig("images/performance_final.png")
plt.savefig("images/performance_final.svg")

plt.show()
