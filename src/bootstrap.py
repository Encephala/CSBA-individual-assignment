#!/usr/bin/env python3

from dataclasses import dataclass
from solution import *

num_hashes = 300

all_divisors = [i for i in range(1, num_hashes + 1) if num_hashes % i == 0]

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

do_print = True

results: dict[float, Result] = {}

for (i, num_rows) in enumerate(all_divisors[:5]):
    print(f"{i + 1} / {len(all_divisors) + 1}: {num_rows} rows")

    num_bands = num_hashes // num_rows
    # Check that num_hashes is divisible by num_rows
    assert num_bands * num_rows == num_hashes

    print(f"(Approximate) LSH Acceptance threshold: {(1 / num_bands) ** (1 / num_rows):.2f}")


    minhash(products, num_hashes, do_print)

    intermediate_duplicates = LSH(products, num_bands, num_rows)

    precision_star, recall_star, F1_star = evaluate(intermediate_duplicates, all_duplicates, num_products, do_print)


    comparison_ratio = len(intermediate_duplicates) / comb(len(products), 2)


    final_duplicates = duplicate_detection(intermediate_duplicates, all_duplicates, do_print)

    precision, recall, F1 = evaluate(final_duplicates, all_duplicates, num_products, do_print)


    results[comparison_ratio] = Result(precision_star, recall_star, F1_star, precision, recall, F1)

print(results)
