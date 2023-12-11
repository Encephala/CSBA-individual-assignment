#!/usr/bin/env python3

# Imports
import json
import warnings

from itertools import combinations
from math import comb
from difflib import SequenceMatcher
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

import numpy as np
import jellyfish

from item import Item, Signature

warnings.filterwarnings("ignore", category = DeprecationWarning)


def preprocess(products: list[Item]) -> None:
    # Calculate weight/diagonal quantiles
    Item.calc_quantiles(products)

    # Find brands for products which it wasn't found yet
    all_brands: set[str] = set()
    for product in products:
        if product.brand:
            all_brands.add(product.brand)

    for product in products:
        if not product.brand:
            for brand in all_brands:
                if brand in product.title:
                    product.brand = brand
                    break

    # Get representation as set
    for product in products:
        product.find_set_representation()


def load_data(filename: str) -> tuple[list[Item], set[tuple[Item]],  int]:
    # Load in data
    with open(filename, "r") as file:
        data = json.load(file)


    # Get all item instances into big array
    products = []

    # Dict which contains modelID -> list of products with that modelID,
    # only for products of which duplicates exist
    duplicates = {}

    for key, val in data.items():
        for product in val:
            model_id = product["modelID"]
            features = product["featuresMap"]
            shop = product["shop"]
            title = product["title"]

            product_as_item = Item(model_id, features, shop, title)

            products.append(product_as_item)

            if model_id not in duplicates:
                duplicates[model_id] = [product_as_item]
            else:
                duplicates[model_id].append(product_as_item)

    # Get set of all duplicates
    all_duplicates: set[tuple[Item]] = set()
    for items in duplicates.values():
        for item, other_item in combinations(items, 2):
            all_duplicates.add((item, other_item))

    num_duplicates = len(all_duplicates)
    print(f"Total number of duplicates: {num_duplicates} / {comb(len(products), 2)}")

    preprocess(products)

    return products, all_duplicates, len(products)


def minhash(products: list[Item], num_hashes: int, do_print: bool = True) -> np.ndarray:
    binary_data = Item.minhash(products, do_print)

    if do_print:
        print("Calculating signatures")
    signatures = Item.binary_to_signatures(binary_data, num_hashes, do_print)

    if do_print:
        print("Done calculating signatures")

    for i, signature in enumerate(signatures.T):
        products[i].signature = Signature(signature)



def LSH(products: list[Item], num_bands: int, num_rows: int) -> set[tuple[Item]]:
    buckets: dict[int, list[Item]] = defaultdict(list)

    for product in products:
        hashes = product.signature.hashes(num_bands, num_rows)
        for subvector_hash in hashes:
            buckets[subvector_hash].append(product)

    # Aggregate buckets
    result: set[tuple[Item]] = set()
    for _, bucket in buckets.items():
        for item, other_item in combinations(bucket, 2):
            result.add((item, other_item))

    return result


def evaluate(found_duplicates: set[tuple[Item]], all_duplicates: set[tuple[Item]], num_products: int, do_print: bool = True) -> list[float]:
    # F1-score
    FP = FN = TP = TN = 0

    # Find TP and FP
    for pair in found_duplicates:
        if pair in all_duplicates:
            TP += 1
        else:
            FP += 1

    # Find FN
    for pair in all_duplicates:
        if pair not in found_duplicates:
            FN += 1

    # Find TN
    TN = comb(num_products, 2) - FN - FP - TP

    if do_print:
        print(f"TP: {TP}")
        print(f"FP: {FP}")
        print(f"TN: {TN}")
        print(f"FN: {FN}")

    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    else:
        precision = 0

    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    else:
        recall = 0


    if precision + recall != 0:
        F1 = 2 * precision * recall / (precision + recall)
    else:
        F1 = 0

    if do_print:
        print(f"F1: {F1:.2%}")

    return precision, recall, F1


def duplicate_detection(intermediate_duplicates: set[tuple[Item]], all_duplicates: set[tuple[Item]], do_print: bool = True) -> set[tuple[Item]]:
    if do_print:
        print("Detecting duplicates")

    def similarity_scores(pair: tuple[Item]):
        item, other_item = pair

        if item.shop == other_item.shop:
            return [0, 0]

        if item.brand != other_item.brand:
            return [0, 0]


        representation, other_representation = [sorted(list(item.set_representation)) for item in pair]
        similarity_SM = SequenceMatcher(None, representation, other_representation).ratio()

        title, other_title = [item.title.replace(" ", "").lower() for item in pair]
        similarity_JW = jellyfish.jaro_winkler_similarity(title, other_title)

        return [similarity_SM, similarity_JW]


    # Fit logit model
    predictor = LogisticRegression().fit(
        [similarity_scores(pair) for pair in intermediate_duplicates],
        [pair in all_duplicates for pair in intermediate_duplicates]
    )

    if do_print:
        print("Done fitting logit model")
        print(f"Coefficients: {predictor.intercept_} {predictor.coef_}")


    final_duplicates: set[tuple[Item]] = set()
    for i, pair in enumerate(intermediate_duplicates):
        if do_print:
            print(f"{i} ({i / len(intermediate_duplicates):.1%})", end = "\r")

        similarity = predictor.predict_proba([similarity_scores(pair)])[0][1]

        if similarity > 0.05:
            # print(f"{similarity_scores(pair)} -> {similarity}")
            final_duplicates.add(pair)

    if do_print:
        print("Done checking duplicates")

    return final_duplicates


if __name__ == "__main__":
    # Parameters
    num_hashes = 420
    num_rows = 3
    num_bands = num_hashes // num_rows
    # Check that num_hashes is divisible by num_rows
    assert num_bands * num_rows == num_hashes

    print(f"(Approximate) LSH Acceptance threshold: {(1 / num_bands) ** (1 / num_rows):.2f}")

    filename = "data/TVs-all-merged.json"


    products, all_duplicates, num_products = load_data(filename)

    print()

    minhash(products, num_hashes)

    intermediate_duplicates = LSH(products, num_bands, num_rows)

    print()

    evaluate(intermediate_duplicates, all_duplicates, num_products)

    print(f"Comparison ratio: {len(intermediate_duplicates) / comb(len(products), 2):.1%}")
    print()

    final_duplicates = duplicate_detection(intermediate_duplicates, all_duplicates)

    evaluate(final_duplicates, all_duplicates, num_products)
