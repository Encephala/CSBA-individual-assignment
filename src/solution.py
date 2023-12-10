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

# Parameters
num_hashes = 300
num_rows = 3
num_bands = num_hashes // num_rows
# Check that num_hashes is divisible by num_rows
assert num_bands * num_rows == num_hashes

print(f"(Approximate) LSH Acceptance threshold: {(1 / num_bands) ** (1 / num_rows):.2f}")

filename = "data/TVs-all-merged.json"

# Load in data
with open(filename, "r") as file:
    data = json.load(file)


# Get all item instances into big array
products = []

# Dict which contains duplicate items
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


# Minhash
binary_data = Item.minhash(products)

print("Calculating signatures")
signatures = Item.binary_to_signatures(binary_data, num_hashes)
print("Done calculating signatures")

for i, signature in enumerate(signatures.T):
    products[i].signature = Signature(signature)


# Locality-sensitive hashing
buckets: dict[int, list[Item]] = defaultdict(list)

for product in products:
    hashes = product.signature.hashes(num_bands, num_rows)
    for subvector_hash in hashes:
        buckets[subvector_hash].append(product)


# F1*-score
FPstar = FNstar = TPstar = TNstar = 0

# Aggregate buckets for FN checking
intermediate_duplicates: set[tuple[Item]] = set()
for _, bucket in buckets.items():
    for item, other_item in combinations(bucket, 2):
        intermediate_duplicates.add((item, other_item))

# Find FP and TP
for pair in intermediate_duplicates:
    if pair in all_duplicates:
        TPstar += 1
    else:
        FPstar += 1

# Find FN
for pair in all_duplicates:
    if pair not in intermediate_duplicates:
        FNstar += 1

# Find TN
TNstar = comb(len(products), 2) - FNstar - FPstar - TPstar


print(f"TP*: {TPstar}")
print(f"FP*: {FPstar}")
print(f"TN*: {TNstar}")
print(f"FN*: {FNstar}")

precision_star = TPstar / (TPstar + FPstar)
recall_star = TPstar / (TPstar + FNstar)

F1_star = 2 * precision_star * recall_star / (precision_star + recall_star)

print(f"F1*: {F1_star:.2%}")

print(f"Comparison ratio: {len(intermediate_duplicates) / comb(len(products), 2):.1%}")


# Robust duplicate detection
print("Detecting duplicates")

# Jaccard score of the sets c1 and c2
def jaccard(pair):
    c1, c2 = [item.set_representation for item in pair]
    return len(c1.intersection(c2)) / len(c1.union(c2))

def similarity_scores(pair: tuple[item]):
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

print("Done fitting logit model")
print(f"Coefficients: {predictor.coef_}")


final_duplicates: set[tuple[Item]] = set()
for i, pair in enumerate(intermediate_duplicates):
    print(f"{i} ({i / len(intermediate_duplicates):.1%})", end = "\r")

    similarity = predictor.predict_proba([similarity_scores(pair)])[0][1]

    if similarity > 0.05:
        # print(f"{similarity_scores(pair)} -> {similarity}")
        final_duplicates.add(pair)

print("Done checking duplicates")


# F1-score
FP = FN = TP = TN = 0

# Find TP and FP
for pair in final_duplicates:
    if pair in all_duplicates:
        TP += 1
    else:
        FP += 1


for pair in all_duplicates:
    if pair not in final_duplicates:
        FN += 1

# Find TN
TN = comb(len(products), 2) - FN - FP - TP


print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"TN: {TN}")
print(f"FN: {FN}")

precision = TP / (TP + FP)
recall = TP / (TP + FN)

F1 = 2 * precision * recall / (precision + recall)

print(f"F1: {F1:.2%}")
