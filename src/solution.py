#!/usr/bin/env python3
import json

import jellyfish

from itertools import combinations
from math import comb
from difflib import SequenceMatcher
from collections import defaultdict

from item import Item, Signature

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

for key, val in data.items():
    for product in val:
        id = product["modelID"]
        features = product["featuresMap"]
        shop = product["shop"]
        title = product["title"]

        products.append(Item(id, features, shop, title))

# Get set of all duplicates
all_duplicates = set()

for key, val in data.items():
    for item, other_item in combinations(val, 2):
        all_duplicates.add((item, other_item))


# Do shingling
for product in products:
    product.shingle(shingle_size)


# Minhash
binary_data = Item.minhash(products)

signatures = Item.binary_to_signatures(binary_data, num_hashes)

for i, signature in enumerate(signatures.T):
    products[i].signature = Signature(signature)


# Locality-sensitive hashing

# A prime significantly larger than the number of products
num_buckets = 15485863

buckets: dict[int, list[Item]] = defaultdict(list)

for product in products:
    hashes = product.signature.hash(num_bands, num_rows)
    for hash in hashes:
        buckets[hash % num_buckets].append(product)



# F1*-score
FP = FN = TP = TN = 0

# Find true and false positives
for _, bucket in buckets.items():
    if len(bucket) > 1:
        for item, other_item in combinations(bucket, 2):
            i += 1
            if item.id == other_item.id:
                TP += 1
            else:
                FP += 1

# This doesn't work yet
for pair in all_duplicates:
    if pair not in buckets.values():
        FN += 1

TN = comb(len(products), 2) - FN - FP - TP

print(f"TP: {TP}")
print(f"FP: {FP}")


num_duplicates = 0
for model, occurrences in data.items():
    length = len(occurrences)
    if length > 1:
        num_duplicates += length * (length - 1) / 2


print(f"Out of: {num_duplicates} duplicates and {comb(len(products), 2):.0f} possible duplicates ({num_duplicates / comb(len(products), 2):.1%})")


detected_duplicates = set()
# Robust duplicate detection
for i, (_, bucket) in enumerate(buckets.items()):
    print(f"{i} ({i / len(buckets):.1%})", end = "\r")
    if len(bucket) > 1:
        for item, other_item in combinations(bucket, 2):
            duplicate = jellyfish.jaro_winkler_similarity(item.title, other_item.title)
            # duplicate = SequenceMatcher(None, item.make_shingle_string(), other_item.make_shingle_string()).ratio()
            if duplicate > 0.9:
                detected_duplicates.add((item, other_item))

print("Done checking duplicates")

# Write duplicates to file for inspection
print(*detected_duplicates, sep = "\n", file = open("/tmp/duplicates.txt", "w"))


# F1-score
FP = FN = TP = TN = 0

# Find true and false positives
for item, other_item in detected_duplicates:
    if item.id == other_item.id:
        TP += 1
    else:
        FP += 1


for pair in all_duplicates:
    if pair not in detected_duplicates:
        FN += 1

TN = comb(len(products), 2) - FN - FP - TP

print(f"TP: {TP}")
print(f"FP: {FP}")


num_duplicates = 0
for model, occurrences in data.items():
    length = len(occurrences)
    if length > 1:
        num_duplicates += length * (length - 1) / 2


print(f"Out of: {len(detected_duplicates)} duplicates and {comb(len(products), 2):.0f} possible duplicates ({len(detected_duplicates) / comb(len(products), 2):.1%})")
