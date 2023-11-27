#!/usr/bin/env python3
# Imports
import json

import jellyfish

from itertools import combinations
from math import comb
from difflib import SequenceMatcher
from collections import defaultdict

from item import Item, Signature


# Parameters
shingle_size = 5

num_hashes = 150
num_bands = 50
num_rows = num_hashes // num_bands
assert num_bands * num_rows == num_hashes

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
# There has to be a better way but this works
all_duplicates = set()
for items in duplicates.values():
    for item, other_item in combinations(items, 2):
        all_duplicates.add((item, other_item))

num_duplicates = len(all_duplicates)
print(f"Total number of duplicates: {num_duplicates} / {comb(len(products), 2)}")
print(all_duplicates, file = open("all_duplicates", "w"))


# Do shingling
for product in products:
    product.shingle(shingle_size)


# Minhash
binary_data = Item.minhash(products)

print("Calculating signatures")
signatures = Item.binary_to_signatures(binary_data, num_hashes)
print("Done calculating signatures")

for i, signature in enumerate(signatures.T):
    products[i].signature = Signature(signature)


# Locality-sensitive hashing
# A prime significantly larger than the number of products
num_buckets = 15485863

buckets: dict[int, list[Item]] = defaultdict(list)

for product in products:
    hashes = product.signature.hash(num_bands, num_rows)
    for subvector_hash in hashes:
        buckets[subvector_hash % num_buckets].append(product)


# F1*-score
FP = FN = TP = TN = 0

# Aggregate buckets for FN checking
intermediate_duplicates = set()
for _, bucket in buckets.items():
    for item, other_item in combinations(bucket, 2):
        intermediate_duplicates.add((item, other_item))

# Find FP and TP
for pair in intermediate_duplicates:
    if pair in all_duplicates:
        TP += 1
    else:
        FP += 1

# Find FN
for pair in all_duplicates:
    if pair not in intermediate_duplicates:
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

print(f"F1*: {F1:.1%}")


# Robust duplicate detection
print("Doing robust duplicate detection")

final_duplicates = set()
for i, pair in enumerate(intermediate_duplicates):
    print(f"{i} ({i / len(intermediate_duplicates):.1%})", end = "\r")

    # similarity = jellyfish.jaro_winkler_similarity(pair[0].title, pair[1].title)
    similarity = SequenceMatcher(None, pair[0].title, pair[1].title).ratio()
    if similarity > 0.7:
        final_duplicates.add(pair)


print("Done checking duplicates")

print(final_duplicates, file = open("final_duplicates", "w"))

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

print(f"F1: {F1:.1%}")
