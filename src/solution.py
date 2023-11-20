#!/usr/bin/env python3
import json

from itertools import combinations

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
num_buckets = 6337

buckets = [[] for i in range(num_buckets)]

for product in products:
    hashes = product.signature.hash(num_bands, num_rows)
    for hash in hashes:
        buckets[hash % num_buckets].append(product)



# Getting some idea of performance
FP = FN = TP = TN = 0

# Find true and false positives
for bucket in buckets:
    if len(bucket) > 1:
        for item, other_item in combinations(bucket, 2):
            i += 1
            if item.id == other_item.id:
                TP += 1
            else:
                FP += 1

print(f"TP: {TP}")
print(f"FP: {FP}")


num_duplicates = 0
for model, occurrences in data.items():
    length = len(occurrences)
    if length > 1:
        num_duplicates += length * (length - 1) / 2


print(f"Out of: {num_duplicates} duplicates")
