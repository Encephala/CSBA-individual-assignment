#!/usr/bin/env python3

import matplotlib.pyplot as plt

from random import choice
from datetime import datetime

from solution import *

num_hashes = 432
weight = 0.8

# Drop highest divisors, as they all yield 0 TP after LSH anyways
all_divisors = [i for i in range(1, num_hashes + 1) if num_hashes % i == 0][:8]
print(f"Testing for num_rows in {all_divisors}")


filename = "data/TVs-all-merged.json"
products, all_duplicates, num_products = load_data(filename)

do_print = False

num_bootstraps = 3

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

    bootstrap_train = set()

    for _ in range(len(products)):
        bootstrap_train.add(choice(tuple(products)))


    bootstrap_test = list(set(products) - bootstrap_train)
    bootstrap_train = list(bootstrap_train)

    all_duplicates_train = set()
    all_duplicates_test = set()

    for product in bootstrap_train:
        for other_product in bootstrap_train:
            if (product, other_product) in all_duplicates:
                all_duplicates_train.add((product, other_product))

    for product in bootstrap_test:
        for other_product in bootstrap_test:
            if (product, other_product) in all_duplicates:
                all_duplicates_test.add((product, other_product))

    num_products_train = len(bootstrap_train)
    num_products_test = len(bootstrap_test)

    print(f"Train: {num_products_train}, test: {num_products_test}")
    print(f"All duplicates train: {len(all_duplicates_train)}, all duplicates test: {len(all_duplicates_test)}", end = "\n\n")


    for (divisor_index, num_rows) in enumerate(all_divisors):
        print(f"{divisor_index + 1} / {len(all_divisors)}: {num_rows} rows")

        num_bands = num_hashes // num_rows
        # Check that num_hashes is divisible by num_rows
        assert num_bands * num_rows == num_hashes

        if do_print:
            print(f"(Approximate) LSH Acceptance threshold: {(1 / num_bands) ** (1 / num_rows):.4f}")


        # Fit model on train data
        if do_print:
            print()
            print("Train data")

        minhash(bootstrap_train, num_hashes, do_print = do_print)

        intermediate_duplicates_train = LSH(bootstrap_train, num_bands, num_rows)

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

        minhash(bootstrap_test, num_hashes, do_print = do_print)

        intermediate_duplicates_test = LSH(bootstrap_test, num_bands, num_rows)

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

# Average results across bootstraps
comparison_ratio = np.mean(results[:, :, 0], axis = 0)
precision_star = np.mean(results[:, :, 1], axis = 0)
recall_star = np.mean(results[:, :, 2], axis = 0)
F1_star = np.mean(results[:, :, 3], axis = 0)
precision = np.mean(results[:, :, 4], axis = 0)
recall = np.mean(results[:, :, 5], axis = 0)
F1 = np.mean(results[:, :, 6], axis = 0)


plt.figure()
plt.title(f"Performance LSH {weight=}")
plt.xlabel("Fraction of comparisons (%)")
plt.ylabel("Performance (%)")
plt.plot(comparison_ratio * 100, precision_star * 100, ".-", label = "Precision*")
plt.plot(comparison_ratio * 100, recall_star * 100, ".-", label = "Recall*")
plt.plot(comparison_ratio * 100, F1_star * 100, ".-", label = "F1*")
plt.legend()
plt.savefig(f"images/performance_LSH {datetime.now()}.png")
plt.savefig(f"images/performance_LSH {datetime.now()}.svg")

plt.figure()
plt.title(f"Performance final {weight=}")
plt.xlabel("Fraction of comparisons (%)")
plt.ylabel("Performance (%)")
plt.plot(comparison_ratio * 100, precision * 100, ".-", label = "Precision*")
plt.plot(comparison_ratio * 100, recall * 100, ".-", label = "Recall*")
plt.plot(comparison_ratio * 100, F1 * 100, ".-", label = "F1*")
plt.legend()
plt.savefig(f"images/performance_final {datetime.now()}.png")
plt.savefig(f"images/performance_final {datetime.now()}.svg")

plt.show()
